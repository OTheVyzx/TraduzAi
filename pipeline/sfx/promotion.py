from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from .inpaint_gate import evaluate_sfx_inpaint_gate
from .mask import build_sfx_glyph_mask
from .style import extract_manhwa_sfx_style


SFX_ROUTE_ACTION = "translate_sfx_inpaint_render"
VISUAL_PROMOTION_THRESHOLD = 0.66
SFX_OCR_OVERLAP_SUPPRESSION_REASON = "visual_sfx_overlap_suppressed"
SFX_OCR_OVERLAP_SUPPRESSION_MIN_RATIO = 0.65
LATIN_SFX_WORDS = {
    "BAM",
    "BANG",
    "BOOM",
    "CRACK",
    "CRASH",
    "CLANG",
    "CLANK",
    "CLICK",
    "GASP",
    "GRAB",
    "GULP",
    "KNOCK",
    "POP",
    "ROAR",
    "RUMBLE",
    "SCREECH",
    "SIGH",
    "SLAM",
    "SNAP",
    "SWOOSH",
    "TAP",
    "THUD",
    "THUMP",
    "WHOOSH",
    "WHAM",
    "ZAP",
}


def promote_visual_sfx_candidate(candidate: dict[str, Any], image_rgb: np.ndarray | None = None) -> dict[str, Any]:
    """Promote visually strong manhwa SFX even when CJK OCR cannot read it.

    This does not grant inpaint/render automatically for unknown text. It marks
    the candidate as a real SFX workflow item and records whether mask evidence
    would allow inpaint after the text/adaptation is supplied.
    """

    result = deepcopy(candidate)
    sfx = result.get("sfx") if isinstance(result.get("sfx"), dict) else {}
    flags = _merge_flags(result.get("qa_flags"), sfx.get("qa_flags"))
    score, score_parts = score_visual_sfx_candidate(result, image_rgb)
    has_text = bool(str(result.get("recognized_text") or result.get("text") or result.get("original") or sfx.get("source_text") or "").strip())
    visually_promoted = score >= VISUAL_PROMOTION_THRESHOLD

    result["content_class"] = "sfx"
    result["tipo"] = "sfx"
    result["script"] = "visual_unknown" if not has_text else str(result.get("script") or "unknown")
    result["sfx_promotion_score"] = round(float(score), 4)
    result["sfx_promotion_score_parts"] = score_parts

    if not visually_promoted:
        result["route_action"] = "review_required"
        result["translate_policy"] = "review"
        result["render_policy"] = "review_required"
        result["route_reason"] = "visual_sfx_score_below_threshold"
        result["qa_flags"] = _merge_flags(flags, ["sfx_visual_candidate", "sfx_visual_promotion_low_score"])
        result["sfx"] = {
            **sfx,
            "visual_promotion": False,
            "promotion_score": round(float(score), 4),
            "promotion_reason": "score_below_threshold",
            "inpaint_allowed": False,
            "qa_flags": _merge_flags(sfx.get("qa_flags"), ["sfx_visual_promotion_low_score"]),
        }
        return result

    result["route_action"] = SFX_ROUTE_ACTION
    result["translate_policy"] = "review" if not has_text else str(result.get("translate_policy") or "adapt_sfx")
    result["render_policy"] = "sfx_style"
    result["route_reason"] = "visual_sfx_promoted_without_ocr" if not has_text else str(result.get("route_reason") or "visual_sfx_promoted")
    result["skip_processing"] = False
    result["preserve_original"] = False
    result["qa_flags"] = _merge_flags(flags, ["sfx_visual_candidate", "sfx_text_unknown"] if not has_text else ["sfx_visual_candidate"])

    mask_evidence, gate, style = _mask_style_gate(result, image_rgb)
    if mask_evidence:
        result["mask_evidence"] = mask_evidence
    if gate:
        result["sfx_inpaint_gate"] = gate
    inpaint_candidate_allowed = bool(gate.get("allow_inpaint")) if isinstance(gate, dict) else False
    inpaint_allowed = bool(inpaint_candidate_allowed and has_text)

    result["sfx"] = {
        **sfx,
        "visual_promotion": True,
        "promotion_score": round(float(score), 4),
        "promotion_reason": "visual_score",
        "source_text": str(result.get("recognized_text") or result.get("text") or result.get("original") or sfx.get("source_text") or "").strip(),
        "adapted_text": str(sfx.get("adapted_text") or result.get("translated") or result.get("traduzido") or "").strip(),
        "translation_mode": str(sfx.get("translation_mode") or ("visual_sfx_manual_text_required" if not has_text else "onomatopoeia_adaptation")),
        "inpaint_candidate_allowed": inpaint_candidate_allowed,
        "inpaint_allowed": inpaint_allowed,
        "style": style or sfx.get("style") or {},
        "qa_flags": _merge_flags(sfx.get("qa_flags"), ["sfx_text_unknown"] if not has_text else []),
    }
    return result


def suppress_normal_ocr_overlapping_sfx(
    texts: list[dict[str, Any]],
    sfx_candidates: list[dict[str, Any]] | None,
    *,
    source_language: str = "en",
) -> list[dict[str, Any]]:
    """Suppress normal OCR layers that are likely misreads of visual SFX.

    This is source-language gated for English works: normal English OCR should
    not translate/inpaint stylized CJK SFX, but clear English dialogue and
    Latin SFX words should remain available.
    """

    if str(source_language or "").strip().lower() not in {"en", "eng", "english"}:
        return [dict(text or {}) for text in texts or []]
    strong_sfx = [candidate for candidate in sfx_candidates or [] if _strong_sfx_overlap_candidate(candidate)]
    if not strong_sfx:
        return [dict(text or {}) for text in texts or []]
    result: list[dict[str, Any]] = []
    for text in texts or []:
        record = dict(text or {})
        text_bbox = _bbox(record.get("bbox") or record.get("text_pixel_bbox"))
        if text_bbox is None:
            result.append(record)
            continue
        match = _best_sfx_overlap(text_bbox, strong_sfx)
        if match is None:
            result.append(record)
            continue
        candidate, overlap = match
        if overlap < SFX_OCR_OVERLAP_SUPPRESSION_MIN_RATIO:
            result.append(record)
            continue
        if _is_latin_sfx_ocr_text(record):
            result.append(_reclassify_latin_ocr_as_sfx(record, candidate, overlap))
            continue
        if _preserve_normal_english_text(record):
            result.append(record)
            continue
        flags = _merge_flags(record.get("qa_flags"), [SFX_OCR_OVERLAP_SUPPRESSION_REASON])
        record["qa_flags"] = flags
        record["skip_processing"] = True
        record["preserve_original"] = False
        record["route"] = "suppress"
        record["route_action"] = "review_required"
        record["route_reason"] = SFX_OCR_OVERLAP_SUPPRESSION_REASON
        record["sfx_overlap_bbox"] = list(_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox")) or [])
        record["sfx_overlap_ratio"] = round(float(overlap), 4)
        record["sfx_candidate_id"] = str(candidate.get("id") or candidate.get("text_id") or "")
        result.append(record)
    return result


def score_visual_sfx_candidate(candidate: dict[str, Any], image_rgb: np.ndarray | None = None) -> tuple[float, dict[str, float]]:
    sfx = candidate.get("sfx") if isinstance(candidate.get("sfx"), dict) else {}
    detector = str(candidate.get("detector") or sfx.get("visual_detector") or "").strip()
    source = str(sfx.get("visual_source") or "").strip()
    confidence = _as_float(candidate.get("confidence", sfx.get("visual_confidence", 0.0)))
    bbox_score = _bbox_score(candidate.get("bbox") or candidate.get("text_pixel_bbox"), image_rgb)
    visual_source_bonus = 0.0
    if detector == "sfx_visual":
        visual_source_bonus = 0.18
    elif source == "anime_text_yolo_low_conf" and confidence >= 0.02:
        visual_source_bonus = 0.12
    elif source == "comic_text_detector_fallback" and confidence >= 0.05:
        visual_source_bonus = 0.10
    crop_score = _crop_text_likeness(candidate, image_rgb)
    ocr_bonus = 0.0
    sfx_ocr = candidate.get("sfx_ocr") if isinstance(candidate.get("sfx_ocr"), dict) else {}
    if str(sfx_ocr.get("status") or "").strip().lower() == "recognized":
        ocr_bonus = 0.20
    score = min(1.0, max(0.0, 0.42 * min(1.0, confidence) + bbox_score + visual_source_bonus + crop_score + ocr_bonus))
    return score, {
        "detector_confidence": round(float(confidence), 4),
        "bbox_score": round(float(bbox_score), 4),
        "visual_source_bonus": round(float(visual_source_bonus), 4),
        "crop_text_likeness": round(float(crop_score), 4),
        "ocr_bonus": round(float(ocr_bonus), 4),
    }


def _strong_sfx_overlap_candidate(candidate: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict):
        return False
    if _bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox")) is None:
        return False
    sfx = candidate.get("sfx") if isinstance(candidate.get("sfx"), dict) else {}
    if bool(sfx.get("visual_promotion")):
        return True
    score = _as_float(candidate.get("sfx_promotion_score", sfx.get("promotion_score", 0.0)))
    if score >= 0.62:
        return True
    confidence = _as_float(candidate.get("confidence", sfx.get("visual_confidence", 0.0)))
    detector = str(candidate.get("detector") or sfx.get("visual_detector") or "").strip()
    source = str(sfx.get("visual_source") or "").strip()
    if detector == "sfx_visual" and confidence >= 0.58:
        return True
    if source == "anime_text_yolo_low_conf" and confidence >= 0.05:
        return True
    if source == "comic_text_detector_fallback" and confidence >= 0.06:
        return True
    return False


def _is_latin_sfx_ocr_text(record: dict[str, Any]) -> bool:
    text = str(record.get("text") or record.get("raw_ocr") or record.get("original") or "").strip()
    if not text:
        return False
    normalized_words = ["".join(ch for ch in word.upper() if ch.isalnum()) for word in text.split()]
    normalized_words = [word for word in normalized_words if word]
    normalized = "".join(ch for ch in text.upper() if ch.isalnum())
    if normalized in LATIN_SFX_WORDS:
        return True
    if normalized_words and all(word in LATIN_SFX_WORDS for word in normalized_words):
        return True
    if len(normalized) >= 4 and normalized.isalpha():
        compact = normalized.rstrip("H")
        if compact in LATIN_SFX_WORDS:
            return True
        for word in LATIN_SFX_WORDS:
            if len(word) >= 4 and normalized[0] == word[0] and normalized[-1] == word[-1]:
                return len(normalized) <= max(10, len(word) + 4)
    return False


def _preserve_normal_english_text(record: dict[str, Any]) -> bool:
    text = str(record.get("text") or record.get("raw_ocr") or record.get("original") or "").strip()
    if not text:
        return False
    normalized = "".join(ch for ch in text.upper() if ch.isalnum())
    words = [word for word in text.replace("'", " ").split() if any(ch.isalpha() for ch in word)]
    if len(words) >= 2:
        short_words = sum(1 for word in words if len(word.strip(".,!?;:")) <= 2)
        return short_words < len(words)
    if any("\u3000" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" for ch in text):
        return False
    if len(normalized) >= 5 and normalized.isalpha():
        vowels = sum(ch in "AEIOU" for ch in normalized)
        return vowels >= 2
    return False


def _reclassify_latin_ocr_as_sfx(record: dict[str, Any], candidate: dict[str, Any], overlap: float) -> dict[str, Any]:
    result = dict(record)
    text = str(result.get("text") or result.get("raw_ocr") or result.get("original") or "").strip()
    sfx = result.get("sfx") if isinstance(result.get("sfx"), dict) else {}
    result["content_class"] = "sfx"
    result["tipo"] = "sfx"
    result["script"] = "latin_sfx"
    result["route_action"] = SFX_ROUTE_ACTION
    result["translate_policy"] = "review"
    result["render_policy"] = "sfx_style"
    result["route_reason"] = "latin_sfx_overlap_visual_candidate"
    result["skip_processing"] = False
    result["preserve_original"] = False
    result["sfx_overlap_bbox"] = list(_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox")) or [])
    result["sfx_overlap_ratio"] = round(float(overlap), 4)
    result["sfx_candidate_id"] = str(candidate.get("id") or candidate.get("text_id") or "")
    result["qa_flags"] = _merge_flags(result.get("qa_flags"), ["latin_sfx_overlap_visual_candidate"])
    result["sfx"] = {
        **sfx,
        "visual_promotion": True,
        "source_text": text,
        "adapted_text": str(sfx.get("adapted_text") or ""),
        "translation_mode": "latin_sfx_manual_adaptation",
        "inpaint_allowed": False,
        "qa_flags": _merge_flags(sfx.get("qa_flags"), ["latin_sfx_overlap_visual_candidate"]),
    }
    return result


def _best_sfx_overlap(text_bbox: list[int], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], float] | None:
    best: tuple[dict[str, Any], float] | None = None
    for candidate in candidates:
        candidate_bbox = _bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"))
        if candidate_bbox is None:
            continue
        overlap = _overlap_against_smaller(text_bbox, candidate_bbox)
        if best is None or overlap > best[1]:
            best = (candidate, overlap)
    return best


def _overlap_against_smaller(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(max(1, min(area_a, area_b)))


def _mask_style_gate(candidate: dict[str, Any], image_rgb: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.size == 0:
        return {}, {}, {}
    mask_result = build_sfx_glyph_mask(image_rgb, candidate)
    result = candidate
    if mask_result.mask is not None and np.any(mask_result.mask):
        result = {**candidate, "mask": mask_result.mask, "mask_evidence": dict(mask_result.evidence)}
    else:
        result = {**candidate, "mask_evidence": dict(mask_result.evidence)}
    gate = evaluate_sfx_inpaint_gate(result)
    style: dict[str, Any] = {}
    bbox = _bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"), image_rgb)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        crop_mask = mask_result.mask[y1:y2, x1:x2] if mask_result.mask is not None else None
        style = extract_manhwa_sfx_style(crop, crop_mask, layer=candidate).to_dict()
    return dict(mask_result.evidence), gate, style


def _crop_text_likeness(candidate: dict[str, Any], image_rgb: np.ndarray | None) -> float:
    bbox = _bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox"), image_rgb)
    if bbox is None or image_rgb is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 55, 150)
    edge_ratio = float(np.mean(edges > 0))
    dark_ratio = float(np.mean(gray <= 90))
    light_ratio = float(np.mean(gray >= 225))
    contrast = float(np.std(gray))
    score = 0.0
    if 0.015 <= edge_ratio <= 0.32:
        score += 0.08
    if 0.01 <= dark_ratio <= 0.55 or 0.01 <= light_ratio <= 0.70:
        score += 0.06
    if contrast >= 28.0:
        score += 0.08
    return min(0.20, score)


def _bbox_score(value: Any, image_rgb: np.ndarray | None) -> float:
    bbox = _bbox(value, image_rgb)
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    if isinstance(image_rgb, np.ndarray) and image_rgb.size:
        page_area = max(1, int(image_rgb.shape[0] * image_rgb.shape[1]))
        ratio = area / float(page_area)
        if 0.004 <= ratio <= 0.18:
            return 0.08
        if 0.001 <= ratio <= 0.30:
            return 0.04
    return 0.04


def _bbox(value: Any, image_rgb: np.ndarray | None = None) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    except Exception:
        return None
    if isinstance(image_rgb, np.ndarray) and image_rgb.size:
        height, width = image_rgb.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _merge_flags(*flag_groups: Any) -> list[str]:
    merged: list[str] = []
    for group in flag_groups:
        for flag in group or []:
            value = str(flag).strip()
            if value and value not in merged:
                merged.append(value)
    return merged
