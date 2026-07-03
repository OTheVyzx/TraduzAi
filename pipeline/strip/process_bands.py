"""Reusa as stages existentes para processar uma Band como se fosse uma página."""

from __future__ import annotations

import copy
import math
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np

from strip.types import Band, BBox
from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask


IMAGE_WHITE_BUBBLE_MASK_SOURCE = "image_white_bubble_mask"
IMAGE_RECT_BUBBLE_MASK_SOURCE = "image_rect_bubble_mask"
IMAGE_CONTOUR_BUBBLE_MASK_SOURCE = "image_contour_bubble_mask"
TEXT_RECT_BUBBLE_MASK_SOURCE = "text_rect_fallback"
ACCEPTED_IMAGE_BUBBLE_MASK_SOURCES = frozenset(
    {
        IMAGE_WHITE_BUBBLE_MASK_SOURCE,
        IMAGE_RECT_BUBBLE_MASK_SOURCE,
        IMAGE_CONTOUR_BUBBLE_MASK_SOURCE,
    }
)

TEXT_RECT_FALLBACK_NOTE_PATTERN = re.compile(r"^\s*T\s*L?\s*/\s*N\s*:", re.IGNORECASE)
WEAK_BUBBLE_MASK_SOURCES = frozenset(
    {
        "derived_white_crop",
        "image_white_region",
        "derived_rectangular_balloon",
        "outline_seeded_contour",
        *ACCEPTED_IMAGE_BUBBLE_MASK_SOURCES,
    }
)
SUPPRESSED_INPAINT_ROUTE_REASONS = frozenset(
    {
        "english_ocr_gibberish_suppressed",
        "false_short_art_ocr",
        "false_unverified_dark_art_ocr",
        "scanlator_text_caption_suppressed",
        "source_language_cjk_text_suppressed",
        "suppressed_duplicate_phrase_fragment",
        "visual_cjk_suppressed",
        "visual_sfx_overlap_suppressed",
    }
)

SOURCE_STYLE_CONFIDENCE_THRESHOLD = 0.70
DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER = 116
DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER = 64


def _style_evidence_confidence(evidence: dict | None) -> float:
    if not isinstance(evidence, dict):
        return 0.0
    fields = []
    if evidence.get("text_color"):
        fields.append("text_color_confidence")
    if evidence.get("stroke_color") or evidence.get("stroke_width_px"):
        fields.append("stroke_confidence")
    if evidence.get("font_name"):
        fields.append("font_confidence")
    if evidence.get("glow") is True:
        fields.append("glow_confidence")
    if evidence.get("shadow") is True:
        fields.append("shadow_confidence")
    values = []
    for field in fields:
        try:
            values.append(float(evidence.get(field) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _hex_luma(value: object) -> float:
    text = str(value or "").strip().lstrip("#")
    if len(text) < 6:
        return 0.0
    try:
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 0.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _rgb_luma_chroma(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        r, g, b = [float(channel) for channel in value[:3]]
    except (TypeError, ValueError):
        return None
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    chroma = max(r, g, b) - min(r, g, b)
    return luma, chroma


def _text_is_visual_card_context(text: dict) -> bool:
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    if source not in {
        "image_white_bubble_mask",
        "image_dark_panel_mask",
        "image_dark_bubble_mask",
        "derived_card_panel_mask",
        "derived_white_crop_rejected",
        "rejected_derived_bubble_mask",
    }:
        return False
    background = _rgb_luma_chroma(text.get("background_rgb"))
    if background is None:
        return False
    luma, chroma = background
    return 95.0 <= luma <= 235.0 and chroma >= 28.0


def _text_is_white_balloon_context(text: dict) -> bool:
    if _text_is_visual_card_context(text):
        return False
    values = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("balloon_type") or "").strip().lower(),
        str(text.get("bubble_mask_source") or "").strip().lower(),
    }
    if values & {"white", "white_balloon", "image_white_bubble_mask", "derived_white_bubble_mask"}:
        return True
    background = _rgb_luma_chroma(text.get("background_rgb"))
    return bool(background is not None and background[0] >= 228.0 and background[1] <= 18.0)


def _source_style_candidate_from_band(text: dict, evidence: dict | None) -> bool:
    if not isinstance(text, dict) or not isinstance(evidence, dict):
        return False
    if _text_is_white_balloon_context(text):
        return False
    if _style_evidence_confidence(evidence) < SOURCE_STYLE_CONFIDENCE_THRESHOLD:
        return False
    text_luma = _hex_luma(evidence.get("text_color"))
    if text_luma < 185.0:
        return False
    background = _rgb_luma_chroma(text.get("background_rgb"))
    if background is None:
        return bool(evidence.get("glow") is True)
    bg_luma, bg_chroma = background
    return bool(bg_luma <= 220.0 or bg_chroma >= 35.0 or evidence.get("glow") is True)


def _style_font_context_for_band_text(text: dict) -> str | None:
    return "visual_card" if _text_is_visual_card_context(text) else None


def _get_band_style_font_detector():
    try:
        from vision_stack.runtime import _get_font_detector

        return _get_font_detector()
    except Exception:
        return None


def _bbox_for_style_crop(text: dict, width: int, height: int) -> list[int] | None:
    bbox = text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    pad = 8
    x1 = max(0, min(width, x1 - pad))
    y1 = max(0, min(height, y1 - pad))
    x2 = max(0, min(width, x2 + pad))
    y2 = max(0, min(height, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _apply_band_source_style_evidence(
    page: dict,
    band_image_bgr: np.ndarray | None,
    *,
    font_detector=None,
) -> int:
    if not isinstance(page, dict) or not isinstance(band_image_bgr, np.ndarray) or band_image_bgr.ndim < 3:
        return 0
    texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
    if not texts:
        return 0
    height, width = band_image_bgr.shape[:2]
    try:
        from typesetter.style_extractor import extract_text_style_evidence
    except Exception:
        return 0
    image_rgb = cv2.cvtColor(band_image_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    resolved_font_detector = font_detector if font_detector is not None else _get_band_style_font_detector()
    applied = 0
    for text in texts:
        if isinstance(text.get("style_evidence"), dict):
            continue
        crop_bbox = _bbox_for_style_crop(text, width, height)
        if crop_bbox is None:
            continue
        x1, y1, x2, y2 = crop_bbox
        try:
            crop = image_rgb[y1:y2, x1:x2, :3]
            kwargs = {}
            if resolved_font_detector is not None:
                kwargs["font_detector"] = resolved_font_detector
            font_context = _style_font_context_for_band_text(text)
            if font_context:
                kwargs["font_context"] = font_context
            evidence_obj = extract_text_style_evidence(crop, **kwargs)
            evidence = evidence_obj.to_dict() if hasattr(evidence_obj, "to_dict") else None
        except Exception:
            continue
        if not _source_style_candidate_from_band(text, evidence):
            if isinstance(evidence, dict):
                text["style_evidence"] = copy.deepcopy(evidence)
                text["style_origin"] = "auto"
                text["style_confidence"] = _style_evidence_confidence(evidence)
            continue
        style = copy.deepcopy(text.get("estilo") or text.get("style") or {})
        style["fonte"] = evidence.get("font_name") or "ComicNeue-Bold.ttf"
        style["cor"] = evidence.get("text_color") or "#F5F5F5"
        style["contorno"] = evidence.get("stroke_color") or ""
        style["contorno_px"] = int(round(float(evidence.get("stroke_width_px") or 0)))
        if evidence.get("glow") is True and float(evidence.get("glow_confidence") or 0.0) >= SOURCE_STYLE_CONFIDENCE_THRESHOLD:
            style["glow"] = True
            style["glow_cor"] = evidence.get("glow_color") or evidence.get("text_color") or "#FFFFFF"
            style["glow_px"] = int(round(float(evidence.get("glow_px") or 2)))
        style["style_origin"] = "source_detected"
        style["style_confidence"] = _style_evidence_confidence(evidence)
        style["style_source"] = evidence.get("source") or "band_style_extractor"
        text["style_evidence"] = copy.deepcopy(evidence)
        text["style"] = copy.deepcopy(style)
        text["estilo"] = copy.deepcopy(style)
        text["style_origin"] = "source_detected"
        text["style_confidence"] = style["style_confidence"]
        text["style_source"] = style["style_source"]
        applied += 1
    return applied


def _accepted_image_bubble_source(source: str | None) -> str | None:
    source_key = str(source or "").strip().lower()
    if source_key in ACCEPTED_IMAGE_BUBBLE_MASK_SOURCES:
        return source_key
    if source_key in {"derived_white_crop", "image_white_region"}:
        return IMAGE_WHITE_BUBBLE_MASK_SOURCE
    if source_key == "derived_rectangular_balloon":
        return IMAGE_RECT_BUBBLE_MASK_SOURCE
    if source_key == "outline_seeded_contour":
        return IMAGE_CONTOUR_BUBBLE_MASK_SOURCE
    return None


_LEGACY_DECISION_FIELDS = frozenset(
    {
        "skip_processing",
        "skip_reason",
        "preserve_original",
        "tipo",
        "content_class",
        "balloon_type",
    }
)


def _legacy_record_key(record: dict, index: int) -> tuple[str, str | int]:
    for key in ("trace_id", "text_id", "id"):
        value = record.get(key)
        if value not in (None, ""):
            return key, str(value)
    bbox = _coerce_bbox(record.get("bbox"))
    if bbox is not None:
        return "bbox", ",".join(str(v) for v in bbox)
    return "index", int(index)


def _legacy_decision_fields_by_record(records) -> dict[tuple[str, str | int], dict]:
    payload: dict[tuple[str, str | int], dict] = {}
    for index, record in enumerate(list(records or [])):
        if not isinstance(record, dict):
            continue
        fields = {
            key: copy.deepcopy(record[key])
            for key in _LEGACY_DECISION_FIELDS
            if record.get(key) not in (None, "")
        }
        if fields:
            payload[_legacy_record_key(record, index)] = fields
    return payload


def _restore_legacy_decision_fields(records, payload: dict[tuple[str, str | int], dict]) -> None:
    if not isinstance(records, list) or not payload:
        return
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        fields = payload.get(_legacy_record_key(record, index)) or payload.get(("index", index))
        if not fields:
            continue
        for key, value in fields.items():
            if record.get(key) in (None, ""):
                record[key] = copy.deepcopy(value)


def _without_legacy_decision_fields_for_stage(page: dict) -> dict:
    stage_page = copy.deepcopy(page or {})
    for list_key in ("texts", "_vision_blocks", "_bubble_regions"):
        records = stage_page.get(list_key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in _LEGACY_DECISION_FIELDS:
                record.pop(key, None)
    return stage_page


def _record_suppressed_for_inpaint(record: dict) -> bool:
    if not isinstance(record, dict):
        return True
    route = str(record.get("route") or record.get("route_action") or "").strip().lower()
    route_reason = str(record.get("route_reason") or "").strip().lower()
    flags = {
        str(flag).strip().lower()
        for flag in (record.get("qa_flags") or [])
        if str(flag).strip()
    }
    return bool(
        route == "suppress"
        or route_reason in SUPPRESSED_INPAINT_ROUTE_REASONS
        or flags & SUPPRESSED_INPAINT_ROUTE_REASONS
    )


def _record_looks_like_false_short_art_ocr(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    qa_flags = {
        str(flag).strip()
        for flag in (record.get("qa_flags") or [])
        if str(flag).strip()
    }
    if not (
        "mask_outside_balloon_critical" in qa_flags
        or "source_glyph_area_ratio_critical" in qa_flags
        or "fast_fill_no_glyph_evidence" in qa_flags
        or record.get("raw_text_evidence_missing")
    ):
        background = record.get("background_rgb")
        background_luma = 255.0
        if isinstance(background, (list, tuple)) and len(background) >= 3:
            try:
                r, g, b = [float(value) for value in background[:3]]
                background_luma = r * 0.299 + g * 0.587 + b * 0.114
            except Exception:
                background_luma = 255.0
        no_line_geometry = not bool(record.get("line_polygons") or record.get("line_polygons_count"))
        if not (no_line_geometry and background_luma <= 245.0):
            return False
    source_text = str(
        record.get("text")
        or record.get("raw_ocr")
        or record.get("original")
        or ""
    ).strip()
    translated = str(record.get("translated") or record.get("traduzido") or "").strip()
    source_compact = re.sub(r"[^A-Za-z0-9]+", "", source_text)
    translated_compact = re.sub(r"[^A-Za-z0-9À-ÖØ-öø-ÿ]+", "", translated)
    if not source_compact or len(source_compact) > 2 or len(translated_compact) > 4:
        return False
    if source_compact.upper() not in {"A", "I", "O"}:
        return False
    bubble_source = str(record.get("bubble_mask_source") or record.get("balloon_mask_source") or "").strip().lower()
    profiles = {
        str(record.get("layout_profile") or "").strip().lower(),
        str(record.get("block_profile") or "").strip().lower(),
        str(record.get("background_type") or "").strip().lower(),
    }
    profiles.discard("")
    return not (bubble_source == "image_dark_bubble_mask" or "dark_bubble" in profiles)


def _mark_false_short_art_ocr_suppressed(record: dict) -> None:
    if not isinstance(record, dict):
        return
    flags = list(record.get("qa_flags") or [])
    if "false_short_art_ocr_suppressed" not in flags:
        flags.append("false_short_art_ocr_suppressed")
    record["qa_flags"] = flags
    record["route_action"] = "review_required"
    record["route_reason"] = "false_short_art_ocr"
    record["skip_processing"] = True
    record["preserve_original"] = True


def _record_is_dark_reocr_fragment(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    source = str(record.get("bubble_mask_source") or record.get("bubbleMaskSource") or "").strip().lower()
    profiles = {
        str(record.get("block_profile") or "").strip().lower(),
        str(record.get("layout_profile") or "").strip().lower(),
    }
    flags = {str(flag).strip() for flag in record.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
        or "dark_bubble" in profiles
        or flags.intersection({"partial_dark_bubble_lobe_reocr", "adjacent_dark_bubble_reocr", "dark_bubble_oval_reocr"})
    )


def _mark_dark_lobe_semantic_duplicate_suppressed(record: dict) -> None:
    flags = list(record.get("qa_flags") or [])
    if "dark_lobe_semantic_duplicate_fragment_suppressed" not in flags:
        flags.append("dark_lobe_semantic_duplicate_fragment_suppressed")
    record["qa_flags"] = flags
    record["route_action"] = "suppress"
    record["route_reason"] = "suppressed_duplicate_phrase_fragment"
    record["skip_processing"] = True
    record["preserve_original"] = True


def _record_looks_like_unverified_dark_art_ocr(record: dict) -> bool:
    if not isinstance(record, dict):
        return False
    flags = {str(flag).strip() for flag in record.get("qa_flags") or [] if str(flag).strip()}
    if not flags.intersection({"dark_bubble_visual_mask_rejected_tiny_text", "debug_derived_bubble_mask_rejected"}):
        return False
    if not flags.intersection({"dark_bubble_lobe_mask_bbox_preferred", "dark_bubble_connected_lobe_passthrough", "dark_bubble_oval_reocr"}):
        return False
    source = str(record.get("bubble_mask_source") or record.get("balloon_mask_source") or "").strip().lower()
    if source in {"image_dark_bubble_mask", "image_dark_panel_mask"}:
        return False
    mask_bbox = _coerce_bbox(record.get("bubble_mask_bbox"))
    if mask_bbox not in (None, [0, 0, 32, 32]):
        return False
    source_text = str(record.get("text") or record.get("original") or record.get("raw_ocr") or "").strip()
    translated = str(record.get("translated") or record.get("traduzido") or "").strip()
    source_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", source_text)
    translated_tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", translated)
    if len(source_tokens) <= 4:
        return True
    if len(source_tokens) <= 6 and len(translated_tokens) <= 3:
        return True
    return False


def _mark_unverified_dark_art_ocr_suppressed(record: dict) -> None:
    flags = list(record.get("qa_flags") or [])
    if "unverified_dark_art_ocr_suppressed" not in flags:
        flags.append("unverified_dark_art_ocr_suppressed")
    record["qa_flags"] = flags
    record["route_action"] = "suppress"
    record["route_reason"] = "false_unverified_dark_art_ocr"
    record["skip_processing"] = True
    record["preserve_original"] = True


def _suppress_dark_lobe_semantic_duplicate_records(texts: list[dict]) -> None:
    def _tokens(record: dict) -> list[str]:
        value = str(record.get("text") or record.get("original") or record.get("raw_ocr") or "").strip()
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9']+", value)
            if len(token) > 1 or token.lower() in {"a", "i"}
        ]

    for candidate in texts:
        if _record_suppressed_for_inpaint(candidate) or not _record_is_dark_reocr_fragment(candidate):
            continue
        candidate_tokens = _tokens(candidate)
        if not (3 <= len(candidate_tokens) <= 8):
            continue
        candidate_bbox = _coerce_bbox(candidate.get("text_pixel_bbox") or candidate.get("bbox") or candidate.get("source_bbox"))
        candidate_balloon = _coerce_bbox(candidate.get("balloon_bbox") or candidate.get("bubble_mask_bbox"))
        for existing in texts:
            if existing is candidate or _record_suppressed_for_inpaint(existing):
                continue
            existing_tokens = _tokens(existing)
            if len(existing_tokens) < len(candidate_tokens) + 2:
                continue
            existing_set = set(existing_tokens)
            shared = sum(1 for token in candidate_tokens if token in existing_set)
            if shared / float(max(1, len(candidate_tokens))) < 0.75:
                continue
            existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
            existing_balloon = _coerce_bbox(existing.get("balloon_bbox") or existing.get("bubble_mask_bbox"))
            text_overlap_ratio = 0.0
            if candidate_bbox is not None and existing_bbox is not None:
                candidate_area = max(1, _bbox_area(candidate_bbox))
                existing_area = max(1, _bbox_area(existing_bbox))
                text_overlap_ratio = _bbox_intersection_area(candidate_bbox, existing_bbox) / float(
                    max(1, min(candidate_area, existing_area))
                )
            balloon_overlap_ratio = 0.0
            if candidate_balloon is not None and existing_balloon is not None:
                candidate_balloon_area = max(1, _bbox_area(candidate_balloon))
                existing_balloon_area = max(1, _bbox_area(existing_balloon))
                balloon_overlap_ratio = _bbox_intersection_area(candidate_balloon, existing_balloon) / float(
                    max(1, min(candidate_balloon_area, existing_balloon_area))
                )
            if text_overlap_ratio >= 0.35 or balloon_overlap_ratio >= 0.35:
                _mark_dark_lobe_semantic_duplicate_suppressed(candidate)
                break


def _drop_suppressed_records_for_inpaint(page: dict) -> None:
    texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
    if not texts:
        page["texts"] = []
        page["_vision_blocks"] = []
        return
    try:
        from ocr.postprocess import apply_language_guards, postprocess_ocr_fragments

        source_language = str(page.get("idioma_origem") or page.get("source_language") or "en")
        guarded_texts = apply_language_guards(
            postprocess_ocr_fragments(texts, page_language=source_language),
            source_language=source_language,
        )
    except Exception:
        guarded_texts = texts

    _suppress_dark_lobe_semantic_duplicate_records(guarded_texts)

    suppressed_text_ids: set[str] = set()
    for text in guarded_texts:
        if _record_looks_like_false_short_art_ocr(text):
            _mark_false_short_art_ocr_suppressed(text)
            text_id = str(text.get("id") or text.get("text_id") or "").strip()
            if text_id:
                suppressed_text_ids.add(text_id)
        elif _record_looks_like_unverified_dark_art_ocr(text):
            _mark_unverified_dark_art_ocr_suppressed(text)
            text_id = str(text.get("id") or text.get("text_id") or "").strip()
            if text_id:
                suppressed_text_ids.add(text_id)

    blocks = [block for block in list(page.get("_vision_blocks") or []) if isinstance(block, dict)]
    if len(blocks) == len(guarded_texts):
        kept_texts: list[dict] = []
        kept_blocks: list[dict] = []
        for text, block in zip(guarded_texts, blocks):
            if _record_suppressed_for_inpaint(text):
                block_id = str(block.get("id") or block.get("text_id") or "").strip()
                if block_id:
                    suppressed_text_ids.add(block_id)
                continue
            kept_texts.append(text)
            kept_blocks.append(block)
        page["texts"] = kept_texts
        page["_vision_blocks"] = kept_blocks
        return

    suppressed_ids = {
        str(text.get("id") or text.get("text_id") or "").strip()
        for text in guarded_texts
        if _record_suppressed_for_inpaint(text)
    }
    suppressed_ids.update(suppressed_text_ids)
    kept_texts = [text for text in guarded_texts if not _record_suppressed_for_inpaint(text)]
    page["texts"] = kept_texts
    if suppressed_ids:
        page["_vision_blocks"] = [
            block
            for block in blocks
            if str(block.get("id") or block.get("text_id") or "").strip() not in suppressed_ids
        ]


def _band_id_for(source_page_number: int | None, band_index: int) -> str:
    try:
        page_number = int(source_page_number or 0)
    except Exception:
        page_number = 0
    return f"page_{max(0, page_number):03d}_band_{max(0, int(band_index)):03d}"


def _page_id_for(source_page_number: int | None) -> str | None:
    try:
        page_number = int(source_page_number or 0)
    except Exception:
        page_number = 0
    if page_number <= 0:
        return None
    return f"page_{page_number:03d}"


def _source_page_number_from_page(page: dict | None, fallback: int | None = None) -> int | None:
    if not isinstance(page, dict):
        return fallback
    for key in ("_source_page_number", "numero", "page_number"):
        value = page.get(key)
        if value is None:
            continue
        try:
            number = int(value)
        except Exception:
            continue
        if number > 0:
            return number
    return fallback


def _coerce_bbox(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if binary.size == 0 or not np.any(binary):
        return binary
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def _edge_contact_ratio(mask: np.ndarray) -> float:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return 0.0
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    contacts = [
        float(np.count_nonzero(mask[0, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[-1, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[:, 0] > 0)) / float(max(1, h)),
        float(np.count_nonzero(mask[:, -1] > 0)) / float(max(1, h)),
    ]
    return max(contacts)


def _edge_contact_side_count(mask: np.ndarray, threshold: float = 0.50) -> int:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return 0
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return 0
    contacts = [
        float(np.count_nonzero(mask[0, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[-1, :] > 0)) / float(max(1, w)),
        float(np.count_nonzero(mask[:, 0] > 0)) / float(max(1, h)),
        float(np.count_nonzero(mask[:, -1] > 0)) / float(max(1, h)),
    ]
    return sum(1 for ratio in contacts if ratio >= threshold)


def _mask_touches_expandable_crop_edge(
    mask: np.ndarray,
    bbox: list[int] | None,
    image_width: int,
    image_height: int,
    threshold: float = 0.02,
) -> bool:
    if not isinstance(mask, np.ndarray) or mask.size == 0 or bbox is None:
        return False
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return False
    x1, y1, x2, y2 = bbox
    top = float(np.count_nonzero(mask[0, :] > 0)) / float(max(1, w))
    bottom = float(np.count_nonzero(mask[-1, :] > 0)) / float(max(1, w))
    left = float(np.count_nonzero(mask[:, 0] > 0)) / float(max(1, h))
    right = float(np.count_nonzero(mask[:, -1] > 0)) / float(max(1, h))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False
    near_margin = max(4, int(round(min(h, w) * 0.025)))
    return bool(
        (top >= threshold and y1 > 0)
        or (bottom >= threshold and y2 < image_height)
        or (left >= threshold and x1 > 0)
        or (right >= threshold and x2 < image_width)
        or (int(ys.min()) <= near_margin and y1 > 0)
        or ((h - int(ys.max()) - 1) <= near_margin and y2 < image_height)
        or (int(xs.min()) <= near_margin and x1 > 0)
        or ((w - int(xs.max()) - 1) <= near_margin and x2 < image_width)
    )


def _dark_border_evidence(crop_rgb: np.ndarray, local_mask: np.ndarray) -> bool:
    if not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return False
    if not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or not np.any(local_mask):
        return False
    gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb.astype(np.uint8)
    ys, xs = np.where(local_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return False
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = 3
    bands: list[np.ndarray] = []
    if y1 > 0:
        bands.append(gray[max(0, y1 - pad) : y1 + 1, max(0, x1 - pad) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[: min(gray.shape[0], pad + 1), :])
    if y2 < gray.shape[0]:
        bands.append(gray[max(0, y2 - 1) : min(gray.shape[0], y2 + pad), max(0, x1 - pad) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[max(0, gray.shape[0] - pad - 1) :, :])
    if x1 > 0:
        bands.append(gray[max(0, y1 - pad) : min(gray.shape[0], y2 + pad), max(0, x1 - pad) : x1 + 1])
    else:
        bands.append(gray[:, : min(gray.shape[1], pad + 1)])
    if x2 < gray.shape[1]:
        bands.append(gray[max(0, y1 - pad) : min(gray.shape[0], y2 + pad), max(0, x2 - 1) : min(gray.shape[1], x2 + pad)])
    else:
        bands.append(gray[:, max(0, gray.shape[1] - pad - 1) :])
    evidence = 0
    for band in bands:
        if band.size == 0:
            continue
        dark_ratio = float(np.count_nonzero(band < 96)) / float(band.size)
        if dark_ratio >= 0.04:
            evidence += 1
    return evidence >= 2


def _classify_derived_bubble_mask(crop_rgb: np.ndarray, local_mask: np.ndarray) -> tuple[str | None, str | None]:
    if not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or not np.any(local_mask):
        return None, "empty_derived_mask"
    h, w = local_mask.shape[:2]
    area = int(np.count_nonzero(local_mask > 0))
    bbox_area = max(1, h * w)
    density = float(area) / float(bbox_area)
    ys, xs = np.where(local_mask > 0)
    component_density = 0.0
    if len(xs) > 0 and len(ys) > 0:
        component_area = max(1, (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))
        component_density = float(area) / float(component_area)
    if component_density >= 0.92 and _dark_border_evidence(crop_rgb, local_mask):
        return "derived_rectangular_balloon", None
    edge_ratio = _edge_contact_ratio(local_mask)
    rectangular_crop = density >= 0.92 and edge_ratio >= 0.92
    if not rectangular_crop:
        return "derived_white_crop", None
    if _dark_border_evidence(crop_rgb, local_mask):
        return "derived_rectangular_balloon", None
    return None, "rejected_rectangular_crop"


def _centered_oval_body_mask_like(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    fitted = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return fitted
    margin_x = max(3, int(round(w * 0.08)))
    margin_y = max(3, int(round(h * 0.10)))
    center = (w // 2, h // 2)
    axes = (max(2, (w - margin_x * 2) // 2), max(2, (h - margin_y * 2) // 2))
    cv2.ellipse(fitted, center, axes, 0, 0, 360, 255, -1)
    return fitted


def _text_support_bbox_for_bubble(block: dict, width: int, height: int) -> list[int] | None:
    bubble_bbox = _coerce_bbox(block.get("bubble_mask_bbox") or block.get("balloon_bbox"))
    for key in (
        "_raw_text_evidence_bbox",
        "raw_text_evidence_bbox",
        "raw_text_bbox",
        "glyph_bbox",
        "ocr_text_bbox",
        "text_pixel_bbox",
        "layout_bbox",
        "bbox",
    ):
        bbox = _coerce_bbox(block.get(key))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            if key == "bbox" and bubble_bbox is not None:
                bbox_area = max(1, (x2 - x1) * (y2 - y1))
                bubble_area = max(1, _bbox_area_value(bubble_bbox))
                overlap = _bbox_intersection_area([x1, y1, x2, y2], bubble_bbox)
                if overlap / float(bbox_area) >= 0.92 and overlap / float(bubble_area) >= 0.80:
                    continue
            return [x1, y1, x2, y2]
    return None


def _bbox_area_value(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _text_rect_fallback_text_value(text: dict) -> str:
    return str(
        text.get("original")
        or text.get("raw_ocr")
        or text.get("recognized_text")
        or text.get("text")
        or text.get("translated")
        or ""
    ).strip()


def _text_rect_fallback_rotation_deg(text: dict) -> float:
    for key in ("rotation_deg", "text_angle_degrees", "angle_deg"):
        try:
            value = float(text.get(key))
        except (TypeError, ValueError):
            continue
        if abs(value) > 0.01:
            normalized = value % 360.0
            if normalized > 180.0:
                normalized -= 360.0
            return float(normalized)
    polygons = text.get("line_polygons")
    if not isinstance(polygons, (list, tuple)):
        return 0.0
    angles: list[float] = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 2:
            continue
        points: list[tuple[float, float]] = []
        for point in polygon[:2]:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if len(points) < 2:
            continue
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))
    if not angles:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=np.float32)))


def _is_text_rect_fallback_candidate(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    if abs(_text_rect_fallback_rotation_deg(text)) >= 5.0:
        return False
    value = _text_rect_fallback_text_value(text)
    content_class = str(text.get("content_class") or text.get("tipo") or "").strip().lower()
    route_reason = str(text.get("route_reason") or text.get("reason") or "").strip().lower()
    flags = {
        str(flag).strip().lower()
        for flag in (text.get("qa_flags") or [])
        if str(flag).strip()
    }
    return bool(
        TEXT_RECT_FALLBACK_NOTE_PATTERN.match(value)
        or content_class in {"tn_note", "translator_note", "caption", "note"}
        or route_reason == "translator_note_marker"
        or "translator_note_marker" in flags
    )


def _text_rect_fallback_support_bbox(text: dict, width: int, height: int) -> list[int] | None:
    candidate_keys = (
        (
            "_raw_text_evidence_bbox",
            "raw_text_evidence_bbox",
            "raw_text_bbox",
            "glyph_bbox",
            "ocr_text_bbox",
            "text_pixel_bbox",
            "layout_bbox",
            "bbox",
        )
        if _is_text_rect_fallback_candidate(text)
        else (
            "_raw_text_evidence_bbox",
            "raw_text_evidence_bbox",
            "raw_text_bbox",
            "glyph_bbox",
            "ocr_text_bbox",
            "text_pixel_bbox",
            "layout_bbox",
            "bbox",
        )
    )
    for key in candidate_keys:
        bbox = _coerce_bbox(text.get(key))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return None


def _apply_text_rect_fallback_mask(text: dict, width: int, height: int, *, padding: int = 4) -> bool:
    if not _is_text_rect_fallback_candidate(text):
        return False
    support_bbox = _text_rect_fallback_support_bbox(text, width, height)
    if support_bbox is None:
        return False
    x1, y1, x2, y2 = support_bbox
    rect_bbox = [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(int(width), x2 + padding),
        min(int(height), y2 + padding),
    ]
    if rect_bbox[2] <= rect_bbox[0] or rect_bbox[3] <= rect_bbox[1]:
        return False
    rect_w = rect_bbox[2] - rect_bbox[0]
    rect_h = rect_bbox[3] - rect_bbox[1]
    text["balloon_bbox"] = copy.deepcopy(rect_bbox)
    text["bubble_mask_bbox"] = copy.deepcopy(rect_bbox)
    text["bubble_inner_bbox"] = copy.deepcopy(rect_bbox)
    text["bubble_mask"] = np.ones((rect_h, rect_w), dtype=np.uint8) * 255
    text["bubble_mask_source"] = TEXT_RECT_BUBBLE_MASK_SOURCE
    text["text_rect_fallback_bbox"] = copy.deepcopy(rect_bbox)
    text["text_rect_fallback_padding_px"] = int(padding)
    text.pop("bubble_mask_error", None)
    return True


def _support_centered_bubble_candidate_bbox(
    support_bbox: list[int] | None,
    width: int,
    height: int,
) -> list[int] | None:
    if support_bbox is None:
        return None
    sx1, sy1, sx2, sy2 = [int(v) for v in support_bbox]
    support_w = max(1, sx2 - sx1)
    support_h = max(1, sy2 - sy1)
    pad_x = max(32, min(96, int(round(support_w * 0.55))))
    pad_y = max(32, min(180, int(round(support_h * 1.20))))
    candidate = [
        max(0, sx1 - pad_x),
        max(0, sy1 - pad_y),
        min(int(width), sx2 + pad_x),
        min(int(height), sy2 + pad_y),
    ]
    if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
        return None
    return candidate


def _derive_tight_text_anchor_bbox_from_image(
    block: dict,
    image_rgb: np.ndarray | None,
    support_bbox: list[int] | None,
) -> list[int] | None:
    if support_bbox is None or not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in support_bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    support_w = max(1, x2 - x1)
    support_h = max(1, y2 - y1)
    support_area = support_w * support_h
    if x2 <= x1 or y2 <= y1 or support_area < 50_000 or (support_w < 220 and support_h < 140):
        return None

    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
    kernel_size = max(7, min(gray.shape[:2]) // 2)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 31)
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0).astype(np.float32)
    gray_f = gray.astype(np.float32)
    dark_contrast = background - gray_f
    light_contrast = gray_f - background
    dark_threshold = max(18.0, min(50.0, float(np.percentile(dark_contrast, 95)) * 0.5))
    light_threshold = max(18.0, min(50.0, float(np.percentile(light_contrast, 95)) * 0.5))
    text_like = ((dark_contrast >= dark_threshold) | (light_contrast >= light_threshold)).astype(np.uint8)
    text_like = cv2.morphologyEx(
        text_like,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(text_like, 8)
    if count <= 1:
        return None

    max_component_area = max(900, int(round(support_area * 0.045)))
    max_component_w = max(90, int(round(support_w * 0.62)))
    max_component_h = max(34, int(round(support_h * 0.28)))
    components: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 16 or area > max_component_area:
            continue
        if cw <= 0 or ch <= 0 or cw > max_component_w or ch > max_component_h:
            continue
        fill = area / float(max(1, cw * ch))
        if fill < 0.08:
            continue
        # Panel/architecture strokes often form long low-fill strips; keep only
        # compact ink clusters as text anchors.
        if (cw >= support_w * 0.45 and ch <= max(6, support_h * 0.035)) or (
            ch >= support_h * 0.45 and cw <= max(6, support_w * 0.035)
        ):
            continue
        components.append((cx, cy, cx + cw, cy + ch, area))
    if not components:
        return None

    components.sort(key=lambda item: (item[1], item[0], -item[4]))
    seed = None
    for component in components:
        cx1, cy1, cx2, cy2, area = component
        if cy1 <= int(round(support_h * 0.42)) or area >= max(120, int(round(support_area * 0.006))):
            seed = component
            break
    if seed is None:
        return None

    sx1, sy1, sx2, sy2, seed_area = seed
    seed_h = max(1, sy2 - sy1)
    seed_w = max(1, sx2 - sx1)
    max_line_gap = max(28, int(round(seed_h * 1.8)), int(round(support_h * 0.08)))
    max_x_gap = max(36, int(round(seed_w * 0.45)), int(round(support_w * 0.10)))
    selected: list[tuple[int, int, int, int, int]] = []
    for component in components:
        cx1, cy1, cx2, cy2, area = component
        center_y = (cy1 + cy2) / 2.0
        if center_y > sy2 + max_line_gap:
            continue
        horizontal_gap = max(0, max(sx1, cx1) - min(sx2, cx2))
        if horizontal_gap > max_x_gap:
            continue
        if area < max(12, int(round(seed_area * 0.02))) and cy1 > sy2:
            continue
        selected.append(component)
    if not selected:
        return None

    ax1 = min(item[0] for item in selected)
    ay1 = min(item[1] for item in selected)
    ax2 = max(item[2] for item in selected)
    ay2 = max(item[3] for item in selected)
    pad_x = max(3, int(round((ax2 - ax1) * 0.025)))
    pad_y = max(3, int(round((ay2 - ay1) * 0.12)))
    anchor = [
        max(0, x1 + ax1 - pad_x),
        max(0, y1 + ay1 - pad_y),
        min(width, x1 + ax2 + pad_x),
        min(height, y1 + ay2 + pad_y),
    ]
    anchor_area = _bbox_area_value(anchor)
    if anchor_area <= 0 or anchor_area >= int(support_area * 0.45):
        return None
    if anchor[2] - anchor[0] < 8 or anchor[3] - anchor[1] < 8:
        return None
    block["_raw_text_evidence_bbox"] = anchor
    block["_raw_text_evidence_pixels"] = int(sum(item[4] for item in selected))
    return anchor


def _derived_bubble_mask_anchor_error(
    local_mask: np.ndarray,
    support_local: tuple[int, int, int, int] | None,
    source: str | None,
) -> str | None:
    if support_local is None or not isinstance(local_mask, np.ndarray) or local_mask.size == 0 or not np.any(local_mask):
        return None
    sx1, sy1, sx2, sy2 = support_local
    support_w = max(1, sx2 - sx1)
    support_h = max(1, sy2 - sy1)
    support_area = max(1, support_w * support_h)
    mask_h, mask_w_full = local_mask.shape[:2]
    sx1 = max(0, min(mask_w_full, sx1))
    sx2 = max(0, min(mask_w_full, sx2))
    sy1 = max(0, min(mask_h, sy1))
    sy2 = max(0, min(mask_h, sy2))
    if sx2 <= sx1 or sy2 <= sy1:
        return "derived_mask_not_anchored_to_text"

    ys, xs = np.where(local_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return "derived_mask_not_anchored_to_text"
    mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    mask_w = max(1, mx2 - mx1)
    mask_h_box = max(1, my2 - my1)
    support_cx = (sx1 + sx2) / 2.0
    support_cy = (sy1 + sy2) / 2.0
    support_hit = int(np.count_nonzero(local_mask[sy1:sy2, sx1:sx2] > 0))
    if support_hit / float(support_area) < 0.08:
        center_probe = cv2.dilate(
            (local_mask > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        cx = max(0, min(mask_w_full - 1, int(round(support_cx))))
        cy = max(0, min(local_mask.shape[0] - 1, int(round(support_cy))))
        if int(center_probe[cy, cx]) == 0:
            return "derived_mask_not_anchored_to_text"

    source_key = str(source or "").strip().lower()
    weak_source = source_key in {"derived_white_crop", "image_white_region", "derived_rectangular_balloon"}
    if not weak_source:
        return None

    area_ratio = (mask_w * mask_h_box) / float(support_area)
    width_ratio = mask_w / float(support_w)
    height_ratio = mask_h_box / float(support_h)
    offset_x = abs(support_cx - (mx1 + mask_w / 2.0)) / float(mask_w)
    offset_y = abs(support_cy - (my1 + mask_h_box / 2.0)) / float(mask_h_box)
    min_margin_x = min(max(0.0, support_cx - mx1), max(0.0, mx2 - support_cx)) / float(mask_w)
    min_margin_y = min(max(0.0, support_cy - my1), max(0.0, my2 - support_cy)) / float(mask_h_box)

    overbroad = area_ratio >= 12.0 and (width_ratio >= 5.0 or height_ratio >= 5.0)
    off_center = max(offset_x, offset_y) >= 0.34 or min(min_margin_x, min_margin_y) <= 0.08
    if overbroad and off_center:
        return "derived_mask_not_anchored_to_text"
    return None


def _weak_existing_bubble_mask_error(
    mask: np.ndarray,
    mask_bbox: list[int] | None,
    support_bbox: list[int] | None,
    source: str | None,
) -> str | None:
    source_key = str(source or "").strip().lower()
    if source_key not in WEAK_BUBBLE_MASK_SOURCES:
        return None
    if not isinstance(mask, np.ndarray) or mask.size == 0 or not np.any(mask):
        return "empty_derived_mask"
    if mask_bbox is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in mask_bbox]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    local = np.where(mask > 0, 255, 0).astype(np.uint8)
    if local.shape[:2] != (target_h, target_w):
        local = cv2.resize(local, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(local > 0)
    if len(xs) == 0 or len(ys) == 0:
        return "empty_derived_mask"
    mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    mask_w = max(1, mx2 - mx1)
    mask_h = max(1, my2 - my1)
    mask_area = int(np.count_nonzero(local > 0))
    mask_bbox_area = max(1, mask_w * mask_h)
    bbox_fill = mask_area / float(mask_bbox_area)

    support_local: tuple[int, int, int, int] | None = None
    support_area = 0
    if support_bbox is not None:
        sx1 = max(0, min(target_w, int(support_bbox[0]) - x1))
        sy1 = max(0, min(target_h, int(support_bbox[1]) - y1))
        sx2 = max(0, min(target_w, int(support_bbox[2]) - x1))
        sy2 = max(0, min(target_h, int(support_bbox[3]) - y1))
        if sx2 > sx1 and sy2 > sy1:
            support_local = (sx1, sy1, sx2, sy2)
            support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
            support_hit = int(np.count_nonzero(local[sy1:sy2, sx1:sx2] > 0))
            if support_hit / float(support_area) < 0.08:
                return "derived_mask_not_anchored_to_text"

    dense_rectangular = (
        source_key in {
            IMAGE_WHITE_BUBBLE_MASK_SOURCE,
            IMAGE_RECT_BUBBLE_MASK_SOURCE,
            "derived_white_crop",
            "image_white_region",
            IMAGE_CONTOUR_BUBBLE_MASK_SOURCE,
            "outline_seeded_contour",
        }
        and (
            bbox_fill >= 0.94
            or (
                source_key in {
                    IMAGE_WHITE_BUBBLE_MASK_SOURCE,
                    IMAGE_RECT_BUBBLE_MASK_SOURCE,
                    "derived_white_crop",
                    "image_white_region",
                }
                and bbox_fill >= 0.86
                and support_area > 0
                and _edge_contact_side_count(local, threshold=0.08) >= 2
            )
            or (
                source_key in {
                    IMAGE_WHITE_BUBBLE_MASK_SOURCE,
                    IMAGE_RECT_BUBBLE_MASK_SOURCE,
                    "derived_white_crop",
                    "image_white_region",
                }
                and bbox_fill >= 0.55
                and support_area > 0
                and mask_bbox_area >= int(max(1, support_area) * 5.0)
                and _edge_contact_side_count(local, threshold=0.08) >= 3
            )
        )
        and mask_area >= max(4096, int(max(1, support_area) * 1.8))
    )
    if dense_rectangular:
        return "suspicious_rectangular_image_bubble_mask"

    if support_local is None:
        return None

    sx1, sy1, sx2, sy2 = support_local
    support_cx = (sx1 + sx2) / 2.0
    support_cy = (sy1 + sy2) / 2.0
    mask_cx = (mx1 + mx2) / 2.0
    mask_cy = (my1 + my2) / 2.0
    offset_x = abs(support_cx - mask_cx) / float(mask_w)
    offset_y = abs(support_cy - mask_cy) / float(mask_h)
    min_margin_x = min(max(0.0, support_cx - mx1), max(0.0, mx2 - support_cx)) / float(mask_w)
    min_margin_y = min(max(0.0, support_cy - my1), max(0.0, my2 - support_cy)) / float(mask_h)
    overbroad_ratio = mask_bbox_area / float(max(1, support_area))
    refined = refine_bubble_shape_mask(local)
    contaminated_irregular = (
        refined.shape_kind in {"irregular", "mixed"}
        and (
            (
                overbroad_ratio >= 18.0
                and (max(offset_x, offset_y) >= 0.28 or min(min_margin_x, min_margin_y) <= 0.12)
            )
            or (
                overbroad_ratio >= 12.0
                and bbox_fill <= 0.72
                and _edge_contact_side_count(local, threshold=0.08) >= 1
            )
            or (
                overbroad_ratio >= 8.0
                and bbox_fill <= 0.75
                and _edge_contact_side_count(local, threshold=0.08) >= 1
                and max(offset_x, offset_y) >= 0.34
            )
            or (
                overbroad_ratio >= 8.0
                and bbox_fill <= 0.70
                and max(offset_x, offset_y) >= 0.24
                and mask_area >= int(max(1, support_area) * 3.0)
            )
        )
    )
    if contaminated_irregular:
        return "contaminated_image_bubble_mask"
    return None


def _tighten_mask_to_nonzero_bbox(mask: np.ndarray, bbox: list[int] | None) -> tuple[np.ndarray, list[int] | None]:
    if bbox is None or not isinstance(mask, np.ndarray) or mask.size == 0 or not np.any(mask):
        return mask, bbox
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return mask, bbox
    lx1, ly1 = int(xs.min()), int(ys.min())
    lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
    if lx2 <= lx1 or ly2 <= ly1:
        return mask, bbox
    tight_bbox = [int(bbox[0]) + lx1, int(bbox[1]) + ly1, int(bbox[0]) + lx2, int(bbox[1]) + ly2]
    return mask[ly1:ly2, lx1:lx2].copy(), tight_bbox


def _prune_mask_to_support_components(
    mask: np.ndarray,
    mask_bbox: list[int] | None,
    support_bbox: list[int] | None,
) -> tuple[np.ndarray | None, list[int] | None, str | None]:
    if (
        mask_bbox is None
        or support_bbox is None
        or not isinstance(mask, np.ndarray)
        or mask.size == 0
        or not np.any(mask)
    ):
        return mask, mask_bbox, None
    x1, y1, x2, y2 = mask_bbox
    target_w = max(1, int(x2) - int(x1))
    target_h = max(1, int(y2) - int(y1))
    local = np.where(mask > 0, 255, 0).astype(np.uint8)
    if local.shape[:2] != (target_h, target_w):
        local = cv2.resize(local, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    sx1 = max(0, min(target_w, int(support_bbox[0]) - int(x1)))
    sy1 = max(0, min(target_h, int(support_bbox[1]) - int(y1)))
    sx2 = max(0, min(target_w, int(support_bbox[2]) - int(x1)))
    sy2 = max(0, min(target_h, int(support_bbox[3]) - int(y1)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None, None, "derived_mask_not_anchored_to_text"

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((local > 0).astype(np.uint8), 8)
    if count <= 1:
        return None, None, "derived_mask_not_anchored_to_text"
    if count == 2:
        return local, mask_bbox, None

    support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    selected: list[int] = []
    best_overlap = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        overlap = int(np.count_nonzero(labels[sy1:sy2, sx1:sx2] == label))
        if overlap <= 0:
            continue
        best_overlap = max(best_overlap, overlap)
        selected.append(label)
    if best_overlap / float(support_area) < 0.08:
        return None, None, "derived_mask_not_anchored_to_text"

    keep = np.zeros_like(local)
    min_overlap = max(1, int(best_overlap * 0.25))
    for label in selected:
        overlap = int(np.count_nonzero(labels[sy1:sy2, sx1:sx2] == label))
        if overlap >= min_overlap:
            keep[labels == label] = 255
    if not np.any(keep):
        return None, None, "derived_mask_not_anchored_to_text"
    keep, tight_bbox = _tighten_mask_to_nonzero_bbox(keep, mask_bbox)
    return keep, tight_bbox, None


def _outline_bubble_mask_around_support(
    crop_rgb: np.ndarray, support_local: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    if support_local is None or not isinstance(crop_rgb, np.ndarray) or crop_rgb.size == 0:
        return None
    crop_h, crop_w = crop_rgb.shape[:2]
    sx1, sy1, sx2, sy2 = support_local
    support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    support_cx = (sx1 + sx2) / 2.0
    support_cy = (sy1 + sy2) / 2.0
    gray = cv2.cvtColor(crop_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if crop_rgb.ndim == 3 else crop_rgb.astype(np.uint8)
    crop_area = max(1, crop_h * crop_w)
    raw_dark = (gray <= 96).astype(np.uint8) * 255
    dark = raw_dark.copy()
    sx1p = max(0, sx1 - 3)
    sy1p = max(0, sy1 - 3)
    sx2p = min(crop_w, sx2 + 3)
    sy2p = min(crop_h, sy2 + 3)
    dark[sy1p:sy2p, sx1p:sx2p] = 0
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2,
    )

    def _mask_from_closed_outline_seed() -> np.ndarray | None:
        barrier = cv2.morphologyEx(
            raw_dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        barrier = cv2.dilate(
            barrier,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        passable = np.where(barrier > 0, 0, 255).astype(np.uint8)
        if passable.size == 0:
            return None
        seed_candidates = [
            (int(round(support_cx)), int(round(support_cy))),
            (sx1 + max(1, (sx2 - sx1) // 4), sy1 + max(1, (sy2 - sy1) // 2)),
            (sx2 - max(1, (sx2 - sx1) // 4), sy1 + max(1, (sy2 - sy1) // 2)),
            (sx1 + max(1, (sx2 - sx1) // 2), sy1 + max(1, (sy2 - sy1) // 4)),
            (sx1 + max(1, (sx2 - sx1) // 2), sy2 - max(1, (sy2 - sy1) // 4)),
        ]
        for sample_y in np.linspace(max(0, sy1 - 6), min(crop_h - 1, sy2 + 6), num=5):
            for sample_x in np.linspace(max(0, sx1 - 6), min(crop_w - 1, sx2 + 6), num=5):
                seed_candidates.append((int(round(float(sample_x))), int(round(float(sample_y)))))
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((passable > 0).astype(np.uint8), 8)
        if count <= 1:
            return None
        best_label = 0
        best_score = -1.0
        for seed_x, seed_y in seed_candidates:
            if seed_x < 0 or seed_y < 0 or seed_x >= crop_w or seed_y >= crop_h:
                continue
            label = int(labels[seed_y, seed_x])
            if label <= 0:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0 or area >= int(crop_area * 0.70):
                continue
            component = labels == label
            support_hit = int(np.count_nonzero(component[sy1:sy2, sx1:sx2]))
            if support_hit < max(8, int(round(support_area * 0.04))):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            edge_ratio = _edge_contact_ratio(np.where(component, 255, 0).astype(np.uint8))
            if edge_ratio >= 0.08:
                continue
            center_distance = abs((x + w / 2.0) - support_cx) / max(1.0, crop_w) + abs(
                (y + h / 2.0) - support_cy
            ) / max(1.0, crop_h)
            score = float(support_hit) * 10_000.0 + float(area) * 0.01 - center_distance * 100.0
            if score > best_score:
                best_score = score
                best_label = label
        if best_label <= 0:
            return None
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        min_support_bridge = max(2, int(round(support_area * 0.005)))
        for label in range(1, count):
            component = labels == label
            support_hit = int(np.count_nonzero(component[sy1:sy2, sx1:sx2]))
            if label != best_label and support_hit < min_support_bridge:
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0 or area >= int(crop_area * 0.70):
                continue
            edge_ratio = _edge_contact_ratio(np.where(component, 255, 0).astype(np.uint8))
            if edge_ratio >= 0.08:
                continue
            mask[component] = 255
        support_bridge = np.zeros_like(mask)
        support_bridge[sy1:sy2, sx1:sx2] = 255
        mask_points = cv2.findNonZero((mask > 0).astype(np.uint8))
        if mask_points is not None and len(mask_points) >= 5:
            hull_mask = np.zeros_like(mask)
            hull = cv2.convexHull(mask_points)
            cv2.fillPoly(hull_mask, [hull], 255)
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(support_bridge, hull_mask))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        mask = _fill_binary_holes(mask)
        if int(np.count_nonzero(mask)) >= int(crop_area * 0.70):
            return None
        if _edge_contact_ratio(mask) >= 0.08:
            return None
        return mask if np.any(mask) else None

    def _model_from_open_outline_components() -> np.ndarray | None:
        component_dark = raw_dark.copy()
        text_component_count, text_labels, text_stats, _text_centroids = cv2.connectedComponentsWithStats(
            (component_dark > 0).astype(np.uint8), 8
        )
        for text_label in range(1, text_component_count):
            area = int(text_stats[text_label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(text_stats[text_label, cv2.CC_STAT_LEFT])
            y = int(text_stats[text_label, cv2.CC_STAT_TOP])
            w = int(text_stats[text_label, cv2.CC_STAT_WIDTH])
            h = int(text_stats[text_label, cv2.CC_STAT_HEIGHT])
            ix1 = max(x, sx1p)
            iy1 = max(y, sy1p)
            ix2 = min(x + w, sx2p)
            iy2 = min(y + h, sy2p)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap_pixels = int(np.count_nonzero(text_labels[iy1:iy2, ix1:ix2] == text_label))
            if overlap_pixels / float(max(1, area)) >= 0.72 and (
                x >= sx1p - 2 and y >= sy1p - 2 and x + w <= sx2p + 2 and y + h <= sy2p + 2
            ):
                component_dark[text_labels == text_label] = 0
        component_dark = cv2.morphologyEx(
            component_dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((component_dark > 0).astype(np.uint8), 8)
        if count <= 1:
            return None
        selected = np.zeros((crop_h, crop_w), dtype=np.uint8)
        support_w = max(1, sx2 - sx1)
        support_h = max(1, sy2 - sy1)
        min_area = max(24, min(180, int(round(support_area * 0.015))))
        selected_sides: set[str] = set()
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if w * h >= int(crop_area * 0.72):
                continue
            if x > sx2 + int(round(support_w * 0.50)):
                continue
            if x + w < sx1 - int(round(support_w * 0.50)):
                continue
            if y > sy2 + int(round(support_h * 0.75)):
                continue
            # Panel dividers and translator notes often sit just below the
            # bubble. They are useful dark pixels, but not part of the speech
            # balloon outline we want to model.
            if y >= sy2 and h <= max(20, int(round(support_h * 0.35))):
                continue
            if y >= sy2 and w >= int(crop_w * 0.18):
                continue
            horizontal_overlap = min(x + w, sx2) - max(x, sx1)
            vertical_overlap = min(y + h, sy2) - max(y, sy1)
            side = None
            encloses_support = (
                x <= sx1 - int(round(support_w * 0.18))
                and x + w >= sx2 + int(round(support_w * 0.18))
                and y <= sy1 - int(round(support_h * 0.35))
                and y + h >= sy2 + int(round(support_h * 0.35))
            )
            if y + h <= support_cy and horizontal_overlap >= -support_w * 0.35:
                side = "top"
            elif x + w <= support_cx and vertical_overlap >= -support_h * 0.35:
                side = "left"
            elif x >= support_cx and vertical_overlap >= -support_h * 0.35:
                side = "right"
            elif y >= support_cy and horizontal_overlap >= int(support_w * 0.15):
                side = "bottom"
            elif encloses_support:
                side = "enclosing"
            if side is None:
                continue
            selected[labels == label] = 255
            selected_sides.add(side)
        if int(np.count_nonzero(selected)) < max(64, min_area * 2):
            return None
        if not (
            "enclosing" in selected_sides
            or {"left", "right"}.issubset(selected_sides)
            or ({"top", "bottom"} & selected_sides)
        ):
            return None
        ys, xs = np.where(selected > 0)
        if len(xs) < 5 or len(ys) < 5:
            return None
        outline_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if not (
            outline_bbox[0] - 8 <= support_cx <= outline_bbox[2] + 8
            and outline_bbox[1] - 8 <= support_cy <= outline_bbox[3] + max(8, support_h)
        ):
            return None
        points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
        hull_model = np.zeros((crop_h, crop_w), dtype=np.uint8)
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillPoly(hull_model, [hull], 255)
        ellipse_model = np.zeros((crop_h, crop_w), dtype=np.uint8)
        try:
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(points)
            axes = (
                max(2, int(round(axis_a * 0.55))),
                max(2, int(round(axis_b * 0.55))),
            )
            cv2.ellipse(
                ellipse_model,
                (int(round(cx)), int(round(cy))),
                axes,
                float(angle),
                0,
                360,
                255,
                -1,
            )
        except cv2.error:
            ellipse_model = hull_model.copy()
        hull_pixels = int(np.count_nonzero(hull_model))
        ellipse_pixels = int(np.count_nonzero(ellipse_model))
        selected_hull_overlap = int(np.count_nonzero((selected > 0) & (hull_model > 0)))
        selected_ellipse_overlap = int(np.count_nonzero((selected > 0) & (ellipse_model > 0)))
        use_hull = bool(
            hull_pixels > 0
            and hull_pixels < int(crop_area * 0.70)
            and selected_hull_overlap >= max(selected_ellipse_overlap, int(np.count_nonzero(selected) * 0.45))
        )
        model = hull_model if use_hull else ellipse_model
        clip_pad_x = max(8, int(round(support_w * 0.18)))
        clip_pad_y = max(8, int(round(support_h * 0.35)))
        clip_pad_y_bottom = max(8, int(round(support_h * 0.12)))
        clip_x1 = max(0, outline_bbox[0] - clip_pad_x)
        clip_y1 = max(0, outline_bbox[1] - clip_pad_y)
        clip_x2 = min(crop_w, outline_bbox[2] + clip_pad_x)
        clip_y2 = min(crop_h, max(outline_bbox[3] + clip_pad_y_bottom, sy2 + 8))
        clip = np.zeros_like(model)
        clip[clip_y1:clip_y2, clip_x1:clip_x2] = 255
        model = cv2.bitwise_and(model, clip)
        support_hit = int(np.count_nonzero(model[sy1:sy2, sx1:sx2] > 0))
        if support_hit / float(max(1, support_area)) < 0.35:
            return None
        if int(np.count_nonzero(model)) >= int(crop_area * 0.72):
            return None
        return model if np.any(model) else None

    seeded_mask = _mask_from_closed_outline_seed()
    if seeded_mask is not None:
        return seeded_mask

    contours, _hierarchy = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_mask: np.ndarray | None = None
    best_score = -1.0
    min_contour_area = max(96.0, min(float(support_area) * 0.35, float(crop_area) * 0.015))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(1, w * h)
        if bbox_area >= int(crop_area * 0.72):
            continue
        if w < max(12, sx2 - sx1) or h < max(12, sy2 - sy1):
            continue
        support_overlap = max(0, min(x + w, sx2) - max(x, sx1)) * max(0, min(y + h, sy2) - max(y, sy1))
        contains_support_center = x <= support_cx <= x + w and y <= support_cy <= y + h
        if support_overlap <= 0 and not contains_support_center:
            continue
        candidate = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.drawContours(candidate, [contour], -1, 255, -1)
        support_hit = int(np.count_nonzero(candidate[sy1:sy2, sx1:sx2] > 0))
        if support_hit <= 0:
            continue
        candidate = _fill_binary_holes(candidate)
        distance = abs((x + w / 2.0) - support_cx) / max(1.0, crop_w) + abs((y + h / 2.0) - support_cy) / max(1.0, crop_h)
        compactness = area / float(bbox_area)
        support_hit_ratio = support_hit / float(max(1, support_area))
        score = support_hit_ratio * 10_000.0 + compactness * 100.0 - distance * 100.0 - area * 0.002
        if score > best_score:
            best_score = score
            best_mask = candidate
    model_mask = _model_from_open_outline_components()
    if best_mask is None or not np.any(best_mask):
        return model_mask
    if model_mask is not None and np.any(model_mask):
        best_pixels = int(np.count_nonzero(best_mask))
        model_pixels = int(np.count_nonzero(model_mask))
        if best_pixels > int(model_pixels * 1.15) or _edge_contact_ratio(best_mask) >= 0.45:
            return model_mask
    return best_mask


def _derive_real_bubble_mask_from_crop(
    image_rgb: np.ndarray | None, bbox: list[int] | None, support_bbox: list[int] | None = None
) -> tuple[np.ndarray | None, str | None, str | None, list[int] | None]:
    def _attempt(
        image_rgb: np.ndarray,
        candidate_bbox: list[int],
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        x1, y1, x2, y2 = candidate_bbox
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, "empty_crop"
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            white = ((gray >= 218) & (val >= 218) & (sat <= 88)).astype(np.uint8) * 255
        else:
            gray = crop.astype(np.uint8)
            white = (gray >= 218).astype(np.uint8) * 255

        crop_area = int(white.size)
        white_pixels = int(np.count_nonzero(white))
        if white_pixels < max(48, int(crop_area * 0.08)):
            return None, None, "not_enough_white_pixels"

        white = cv2.morphologyEx(
            white,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        white = _fill_binary_holes(white)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
        if count <= 1:
            return None, None, "no_white_component"

        crop_h, crop_w = white.shape[:2]
        center_x = crop_w / 2.0
        center_y = crop_h / 2.0
        support_local: tuple[int, int, int, int] | None = None
        if support_bbox is not None:
            sx1 = max(0, min(crop_w, int(support_bbox[0]) - x1))
            sy1 = max(0, min(crop_h, int(support_bbox[1]) - y1))
            sx2 = max(0, min(crop_w, int(support_bbox[2]) - x1))
            sy2 = max(0, min(crop_h, int(support_bbox[3]) - y1))
            if sx2 > sx1 and sy2 > sy1:
                support_local = (sx1, sy1, sx2, sy2)

        best_label = 0
        best_score = -1.0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(48, int(crop_area * 0.06)):
                continue
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            support_overlap = 0
            if support_local is not None:
                sx1, sy1, sx2, sy2 = support_local
                ix1 = max(left, sx1)
                iy1 = max(top, sy1)
                ix2 = min(left + comp_w, sx2)
                iy2 = min(top + comp_h, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    support_overlap = int(np.count_nonzero(labels[iy1:iy2, ix1:ix2] == label))
                if support_overlap <= 0:
                    continue
            comp_cx = left + comp_w / 2.0
            comp_cy = top + comp_h / 2.0
            center_distance = abs(comp_cx - center_x) / max(1.0, crop_w) + abs(comp_cy - center_y) / max(1.0, crop_h)
            score = float(area) - center_distance * float(crop_area) * 0.25
            if support_overlap > 0:
                comp_bbox_area = max(1, comp_w * comp_h)
                compactness = float(area) / float(comp_bbox_area)
                support_ratio = float(support_overlap) / float(max(1, (support_local[2] - support_local[0]) * (support_local[3] - support_local[1])))
                score = support_overlap * 10_000.0 + compactness * 1_000.0 + support_ratio * 1_000.0 - float(area) * 0.01
            if score > best_score:
                best_score = score
                best_label = label
        if best_label <= 0:
            return None, None, "no_supported_white_component"

        local = np.where(labels == best_label, 255, 0).astype(np.uint8)
        local = cv2.morphologyEx(
            local,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        local = _fill_binary_holes(local)
        source_override: str | None = None
        if not np.any(local):
            return None, None, "empty_derived_mask"
        local_density = float(np.count_nonzero(local > 0)) / float(max(1, local.size))
        preliminary_source, _preliminary_error = _classify_derived_bubble_mask(crop, local)
        if support_local is not None and (
            local_density >= 0.55
            or _edge_contact_side_count(local, threshold=0.20) >= 2
            or preliminary_source == "derived_rectangular_balloon"
        ):
            outline_mask = _outline_bubble_mask_around_support(crop, support_local)
            if outline_mask is not None:
                local = outline_mask
                source_override = "outline_seeded_contour"
                local_density = float(np.count_nonzero(local > 0)) / float(max(1, local.size))
                preliminary_source, _preliminary_error = _classify_derived_bubble_mask(crop, local)
        if (
            support_local is None
            and local_density >= 0.75
            and _edge_contact_ratio(local) >= 0.80
            and _edge_contact_side_count(local) >= 2
            and preliminary_source == "derived_white_crop"
        ):
            local = _centered_oval_body_mask_like(local)
        if source_override != "outline_seeded_contour":
            local = refine_bubble_shape_mask(local).mask
        source, error = _classify_derived_bubble_mask(crop, local)
        if source is None:
            return None, None, error
        anchor_error = _derived_bubble_mask_anchor_error(local, support_local, source_override or source)
        if anchor_error is not None:
            return None, None, anchor_error
        return local, source_override or source, None

    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None, None, "missing_image", None
    bbox = _coerce_bbox(bbox)
    if bbox is None:
        return None, None, "missing_bbox", None
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None, "invalid_bbox", None

    candidate_bbox = [x1, y1, x2, y2]
    mask, source, error = _attempt(image_rgb, candidate_bbox)
    should_retry_expanded = bool(
        error == "rejected_rectangular_crop"
        or (
            mask is not None
            and source == "outline_seeded_contour"
            and _mask_touches_expandable_crop_edge(mask, candidate_bbox, width, height)
        )
        or (
            mask is not None
            and source == "derived_rectangular_balloon"
            and _edge_contact_ratio(mask) >= 0.80
        )
        or (
            mask is not None
            and source == "derived_white_crop"
            and _edge_contact_ratio(mask) >= 0.50
        )
    )
    if mask is not None and not should_retry_expanded:
        return mask, source, None, candidate_bbox
    if not should_retry_expanded:
        return None, None, error, None

    for margin in (16, 24, 48, 80, 120, 160):
        expanded = [
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(width, x2 + margin),
            min(height, y2 + margin),
        ]
        if expanded == candidate_bbox:
            continue
        expanded_mask, expanded_source, expanded_error = _attempt(image_rgb, expanded)
        if expanded_mask is None:
            continue
        if expanded_source == "derived_rectangular_balloon":
            continue
        if (
            expanded_source == "outline_seeded_contour"
            and _mask_touches_expandable_crop_edge(expanded_mask, expanded, width, height)
        ):
            continue
        return expanded_mask, expanded_source, None, expanded

    if mask is not None:
        return mask, source, None, candidate_bbox
    return None, None, error, candidate_bbox


def _attach_real_bubble_mask_to_block(block: dict, image_rgb: np.ndarray | None) -> None:
    if not isinstance(block, dict):
        return
    has_image = isinstance(image_rgb, np.ndarray) and image_rgb.size > 0
    height = int(image_rgb.shape[0]) if has_image else 0
    width = int(image_rgb.shape[1]) if has_image else 0
    existing = block.get("bubble_mask")
    invalid_existing_weak_mask = False
    if isinstance(existing, np.ndarray) and existing.size > 0 and np.any(existing):
        source = str(block.get("bubble_mask_source") or block.get("bubbleMaskSource") or "").strip().lower()
        if source in {"real", "real_bubble_mask", "speech_bubble_segmentation", "speech-bubble-segmentation"}:
            block["bubble_mask_source"] = "real_bubble_mask"
            block.pop("bubble_mask_error", None)
            return
        if source in {
            "image_white_bubble_mask",
            "image_rect_bubble_mask",
            "image_contour_bubble_mask",
            "derived_white_crop",
            "derived_rectangular_balloon",
            "outline_seeded_contour",
        }:
            support_bbox = _text_support_bbox_for_bubble(block, width, height) if has_image else None
            tight_support_bbox = _derive_tight_text_anchor_bbox_from_image(block, image_rgb, support_bbox) if has_image else None
            if tight_support_bbox is not None:
                support_bbox = tight_support_bbox
            mask_bbox = _coerce_bbox(block.get("bubble_mask_bbox") or block.get("balloon_bbox") or block.get("bbox"))
            weak_error = _weak_existing_bubble_mask_error(existing, mask_bbox, support_bbox, source)
            if weak_error is None:
                return
            block.pop("bubble_mask", None)
            block.pop("bubble_inner_bbox", None)
            block["bubble_mask_source"] = "derived_white_crop_rejected"
            block["bubble_mask_error"] = weak_error
            invalid_existing_weak_mask = True
            if not has_image:
                return
        else:
            block["bubble_mask_source"] = "bbox_fallback"
            block["bubble_mask_error"] = "missing_real_bubble_mask_source"
            return
    if not has_image:
        block.setdefault("bubble_mask_source", "bbox_fallback")
        block.setdefault("bubble_mask_error", "missing_real_bubble_mask")
        return
    support_bbox = _text_support_bbox_for_bubble(block, width, height)
    tight_support_bbox = _derive_tight_text_anchor_bbox_from_image(block, image_rgb, support_bbox)
    if tight_support_bbox is not None:
        support_bbox = tight_support_bbox
    bbox_candidates: list[list[int]] = []
    for raw in (block.get("balloon_bbox"), block.get("bubble_mask_bbox"), block.get("bbox")):
        candidate = _coerce_bbox(raw)
        if candidate is None:
            continue
        if any(candidate == existing for existing in bbox_candidates):
            continue
        bbox_candidates.append(candidate)
    if support_bbox is not None and bbox_candidates:
        support_area = max(1, _bbox_area_value(support_bbox))
        bbox_candidates.sort(key=lambda candidate: (_bbox_area_value(candidate) > support_area * 18, _bbox_area_value(candidate)))
    if support_bbox is not None and (
        invalid_existing_weak_mask
        or (
            bbox_candidates
            and _bbox_area_value(bbox_candidates[0]) > max(4096, _bbox_area_value(support_bbox) * 6)
        )
    ):
        support_candidate = _support_centered_bubble_candidate_bbox(support_bbox, width, height)
        if support_candidate is not None and not any(support_candidate == existing for existing in bbox_candidates):
            bbox_candidates.insert(0, support_candidate)
    mask = None
    source = None
    error = "missing_bbox"
    used_bbox = None
    bbox = bbox_candidates[0] if bbox_candidates else None
    for candidate_index, candidate in enumerate(bbox_candidates):
        mask, source, error, used_bbox = _derive_real_bubble_mask_from_crop(image_rgb, candidate, support_bbox)
        if mask is not None:
            fresh_weak_error = _weak_existing_bubble_mask_error(mask, used_bbox, support_bbox, source)
            if fresh_weak_error is not None:
                mask = None
                source = None
                error = fresh_weak_error
                used_bbox = None
                continue
            if (
                not invalid_existing_weak_mask
                and support_bbox is not None
                and used_bbox is not None
                and candidate_index + 1 < len(bbox_candidates)
                and _bbox_area_value(candidate) <= _bbox_area_value(support_bbox) * 4
                and _bbox_area_value(used_bbox) > int(_bbox_area_value(candidate) * 2.5)
            ):
                continue
            bbox = candidate
            break
    if mask is not None:
        accepted_source = _accepted_image_bubble_source(source)
        if (
            source == "derived_white_crop"
            and used_bbox is not None
            and bbox is not None
            and used_bbox != bbox
            and bbox[0] >= used_bbox[0]
            and bbox[1] >= used_bbox[1]
            and bbox[2] <= used_bbox[2]
            and bbox[3] <= used_bbox[3]
            and _bbox_area_value(bbox) >= int(_bbox_area_value(used_bbox) * 0.55)
        ):
            lx1 = int(bbox[0] - used_bbox[0])
            ly1 = int(bbox[1] - used_bbox[1])
            lx2 = int(bbox[2] - used_bbox[0])
            ly2 = int(bbox[3] - used_bbox[1])
            candidate_crop = image_rgb[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            candidate_mask = mask[ly1:ly2, lx1:lx2].copy()
            if (
                candidate_crop.size > 0
                and candidate_mask.size > 0
                and np.any(candidate_mask)
            ):
                mask = candidate_mask
                used_bbox = bbox
        outline_source_can_tighten = bool(
            source == "outline_seeded_contour"
            and support_bbox is not None
            and used_bbox is not None
            and _bbox_area_value(used_bbox) > _bbox_area_value(support_bbox) * 18
        )
        if support_bbox is not None and used_bbox is not None and (
            source != "derived_rectangular_balloon"
            and (source != "outline_seeded_contour" or outline_source_can_tighten)
        ):
            tight_mask, tight_bbox = _tighten_mask_to_nonzero_bbox(mask, used_bbox)
            if tight_bbox is not None and _bbox_area_value(tight_bbox) <= int(_bbox_area_value(used_bbox) * 0.72):
                mask = tight_mask
                used_bbox = tight_bbox
        if used_bbox is not None and (used_bbox != bbox or source != "derived_rectangular_balloon"):
            block["bubble_mask_bbox"] = used_bbox
        block["bubble_mask"] = mask
        block["bubble_mask_source"] = accepted_source or source or IMAGE_WHITE_BUBBLE_MASK_SOURCE
        block.pop("bubble_mask_error", None)
    else:
        bubble_error = "rejected_rectangular_crop" if error == "rejected_rectangular_crop" else "missing_real_bubble_mask"
        if error == "rejected_rectangular_crop":
            block["bubble_mask_source"] = "derived_white_crop_rejected"
            block["bubble_mask_error"] = bubble_error
        else:
            block.setdefault("bubble_mask_source", "bbox_fallback")
            block.setdefault("bubble_mask_error", bubble_error)


def _contract_value_present(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > 0 and bool(np.any(value))
    return value not in (None, [], "")


def _contract_value_missing(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0 or not bool(np.any(value))
    return value in (None, [], "")


def _has_rejected_bubble_mask_contract(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    error = str(text.get("bubble_mask_error") or "").strip().lower()
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    return error == "rejected_rectangular_crop" or source == "derived_white_crop_rejected"


def _shift_bbox_y(value, delta_y: int) -> list[int] | None:
    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    return [bbox[0], bbox[1] + int(delta_y), bbox[2], bbox[3] + int(delta_y)]


def _shift_bbox_xy(value, delta_x: int, delta_y: int) -> list[int] | None:
    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    return [
        bbox[0] + int(delta_x),
        bbox[1] + int(delta_y),
        bbox[2] + int(delta_x),
        bbox[3] + int(delta_y),
    ]


def _confidence_value(*values) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return round(float(value), 4)
        except Exception:
            continue
    return None


def _text_id_for(text: dict, index: int) -> str:
    return str(text.get("text_id") or text.get("id") or text.get("_id") or f"ocr_{index + 1:03d}")


def _trace_id_for(text_id: str, band_id: str) -> str:
    return f"{text_id}@{band_id}" if band_id else text_id


def _unique_string_list(value) -> list[str]:
    values: list[str] = []
    if isinstance(value, str):
        iterable = [value]
    else:
        iterable = value or []
    for item in iterable:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _attach_source_trace_metadata(record: dict, *, band_id: str) -> None:
    source_text_ids = _unique_string_list(
        record.get("source_text_ids") or record.get("_source_text_ids")
    )
    source_trace_ids = _unique_string_list(
        record.get("source_trace_ids") or record.get("_source_trace_ids")
    )
    for source_text_id in source_text_ids:
        trace_id = _trace_id_for(source_text_id, band_id)
        if trace_id not in source_trace_ids:
            source_trace_ids.append(trace_id)
    if source_text_ids:
        record["source_text_ids"] = source_text_ids
    if source_trace_ids:
        record["source_trace_ids"] = source_trace_ids
        record["_source_trace_ids"] = source_trace_ids
    try:
        merged_count = int(record.get("ocr_merged_source_count") or 0)
    except Exception:
        merged_count = 0
    if (merged_count > 1 or len(source_text_ids) > 1 or len(source_trace_ids) > 1) and not record.get("merge_reason"):
        record["merge_reason"] = "clustered_line_fragments"


def _trace_ids_from_page(page: dict | None, *, band_id: str) -> list[str]:
    if not isinstance(page, dict):
        return []
    trace_ids: list[str] = []
    for index, text in enumerate(list(page.get("texts") or [])):
        if not isinstance(text, dict):
            continue
        text_band_id = str(text.get("band_id") or band_id or "")
        if band_id and text_band_id and text_band_id != band_id:
            continue
        text_id = _text_id_for(text, index)
        trace_id = str(text.get("trace_id") or _trace_id_for(text_id, text_band_id or band_id))
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)
    for index, block in enumerate(list(page.get("_vision_blocks") or [])):
        if not isinstance(block, dict):
            continue
        block_band_id = str(block.get("band_id") or band_id or "")
        if band_id and block_band_id and block_band_id != band_id:
            continue
        block_text_id_raw = block.get("text_id") or block.get("id")
        if not block_text_id_raw:
            continue
        block_text_id = str(block_text_id_raw)
        trace_id = str(block.get("trace_id") or _trace_id_for(block_text_id, block_band_id or band_id))
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)
    return trace_ids


def _trace_metadata_payload(page: dict | None, *, band_id: str, source_page_number: int | None = None) -> dict:
    page_number = _source_page_number_from_page(page, source_page_number)
    page_id = _page_id_for(page_number)
    trace_ids = _trace_ids_from_page(page, band_id=band_id)
    text_ids = []
    if isinstance(page, dict):
        for index, text in enumerate(list(page.get("texts") or [])):
            if not isinstance(text, dict):
                continue
            text_band_id = str(text.get("band_id") or band_id or "")
            if band_id and text_band_id and text_band_id != band_id:
                continue
            text_id = _text_id_for(text, index)
            if text_id not in text_ids:
                text_ids.append(text_id)
    payload = {
        "page_id": page_id,
        "band_id": band_id,
        "text_ids": text_ids,
        "trace_ids": trace_ids,
        "trace_ids_in_band": trace_ids,
    }
    if len(text_ids) == 1:
        payload["text_id"] = text_ids[0]
    return {key: value for key, value in payload.items() if value not in (None, [], "")}


def _attach_ocr_trace_metadata(page: dict, *, band_id: str) -> dict:
    if not isinstance(page, dict) or not band_id:
        return page
    page_number = _source_page_number_from_page(page)
    page_id = _page_id_for(page_number)
    texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
    blocks = [block for block in list(page.get("_vision_blocks") or []) if isinstance(block, dict)]
    seen_text_ids: dict[str, int] = {}
    for index, text in enumerate(texts):
        base_text_id = _text_id_for(text, index)
        seen_count = seen_text_ids.get(base_text_id, 0)
        seen_text_ids[base_text_id] = seen_count + 1
        text_id = base_text_id if seen_count == 0 else f"{base_text_id}_{seen_count + 1:03d}"
        if text_id != base_text_id:
            text.setdefault("_original_text_id", base_text_id)
            if text.get("trace_id"):
                text.setdefault("_original_trace_id", str(text.get("trace_id")))
            text.pop("trace_id", None)
            text.pop("text_instance_id", None)
        text["id"] = text_id
        text["text_id"] = text_id
        text["band_id"] = band_id
        if page_id:
            text["page_id"] = page_id
        text["trace_id"] = str(text.get("trace_id") or _trace_id_for(text_id, band_id))
        _attach_source_trace_metadata(text, band_id=band_id)
        confidence_raw = _confidence_value(text.get("confidence_raw"), text.get("confidence"), text.get("ocr_confidence"))
        if confidence_raw is not None:
            text["confidence_raw"] = confidence_raw
        if index < len(blocks):
            block = blocks[index]
            block["text_id"] = text_id
            block["band_id"] = band_id
            if page_id:
                block["page_id"] = page_id
            block["trace_id"] = text["trace_id"]
            for key in (
                "source_text_ids",
                "source_trace_ids",
                "_source_trace_ids",
                "_merged_source_bboxes",
                "merged_source_bboxes",
                "merge_reason",
                "rotation_deg",
                "rotation_source",
                "qa_flags",
                "allow_broad_bbox_text_search",
                "balloon_type",
                "block_profile",
                "line_polygons",
                "text_pixel_bbox",
                "source_bbox",
                "balloon_bbox",
                "background_rgb",
                "layout_bbox",
                "content_class",
                "skip_processing",
                "preserve_original",
                "render_policy",
                "route_action",
                "route_reason",
                "bubble_mask_source",
                "bubbleMaskSource",
                "bubble_mask_bbox",
                "bubbleMaskBbox",
                "bubble_inner_bbox",
                "bubbleInnerBbox",
                "bubble_id",
                "bubbleId",
                "bubble_mask_error",
                "bubbleMaskError",
                "card_panel_text_context",
                "mask_evidence",
            ):
                if text.get(key) not in (None, [], ""):
                    block[key] = copy.deepcopy(text[key])
            block_confidence_raw = _confidence_value(
                block.get("confidence_raw"),
                block.get("confidence"),
                confidence_raw,
            )
            if block_confidence_raw is not None:
                block["confidence_raw"] = block_confidence_raw
    page["texts"] = texts
    page["_vision_blocks"] = blocks
    page["_band_id"] = band_id
    if page_id:
        page["_page_id"] = page_id
    page["_trace_ids_in_band"] = _trace_ids_from_page(page, band_id=band_id)
    return page


def _record_ocr_raw_blocks(page: dict, *, band: Band, band_id: str) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        texts = [text for text in list((page or {}).get("texts") or []) if isinstance(text, dict)]
        blocks = [block for block in list((page or {}).get("_vision_blocks") or []) if isinstance(block, dict)]
        page_id = _page_id_for(_source_page_number_from_page(page))
        for index, text in enumerate(texts):
            text_id = _text_id_for(text, index)
            block = blocks[index] if index < len(blocks) else {}
            confidence_raw = _confidence_value(
                text.get("confidence_raw"),
                block.get("confidence_raw") if isinstance(block, dict) else None,
                text.get("confidence"),
                block.get("confidence") if isinstance(block, dict) else None,
            )
            bbox_band = _coerce_bbox(text.get("bbox"))
            source_bbox_band = _coerce_bbox(text.get("source_bbox"))
            text_pixel_bbox_band = _coerce_bbox(text.get("text_pixel_bbox"))
            line_polygons = text.get("line_polygons") or []
            payload = {
                "text_id": text_id,
                "page_id": page_id,
                "band_id": band_id,
                "trace_id": str(text.get("trace_id") or _trace_id_for(text_id, band_id)),
                "raw_ocr": text.get("raw_ocr") or text.get("original") or text.get("text") or "",
                "confidence_raw": confidence_raw,
                "bbox_band": bbox_band,
                "bbox_page": _shift_bbox_y(bbox_band, int(band.y_top)) if bbox_band else None,
                "text_pixel_bbox_band": text_pixel_bbox_band,
                "source_bbox_band": source_bbox_band,
                "line_polygons_count": len(line_polygons) if isinstance(line_polygons, list) else 0,
                "background_rgb": text.get("background_rgb"),
                "balloon_type": text.get("balloon_type"),
                "block_profile": text.get("block_profile"),
                "accepted": True,
                "accept_reason": "ready_for_layout",
                "reject_reason": None,
                "ocr_backend": text.get("ocr_source") or text.get("ocr_mode"),
            }
            recorder.write_jsonl(
                "03_ocr/ocr_raw_blocks.jsonl",
                {key: value for key, value in payload.items() if value is not None},
            )
    except Exception:
        return


def _record_copyback_decision(
    *,
    band: Band,
    band_id: str,
    source_page_number: int | None,
    translated_page: dict | None,
    applied: bool,
    reason: str,
) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        texts = [
            text
            for text in list((translated_page or {}).get("texts") or [])
            if isinstance(text, dict)
        ]
        recorder.write_jsonl(
            "10_copyback_reassemble/copyback_decisions.jsonl",
            {
                "band_id": band_id,
                "source_page_number": int(source_page_number or 0),
                "y_top": int(getattr(band, "y_top", 0) or 0),
                "y_bottom": int(getattr(band, "y_bottom", 0) or 0),
                "balloon_count": int(len(getattr(band, "balloons", []) or [])),
                "text_count": int(len(texts)),
                "copyback_applied": bool(applied),
                "reason": str(reason),
                **_trace_metadata_payload(
                    translated_page,
                    band_id=band_id,
                    source_page_number=source_page_number,
                ),
            },
        )
    except Exception:
        return


def _record_band_stage_visual_debug(
    *,
    band: Band,
    band_id: str,
    source_page_number: int | None,
    post_typeset: np.ndarray | None,
    post_copyback: np.ndarray | None,
) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        post_typeset_path = f"09_typeset/{band_id}/post_typeset.jpg"
        post_copyback_path = f"10_copyback_reassemble/{band_id}/post_copyback.jpg"
        if isinstance(post_typeset, np.ndarray) and post_typeset.size:
            recorder.write_image(post_typeset_path, post_typeset, quality=92)
        if isinstance(post_copyback, np.ndarray) and post_copyback.size:
            recorder.write_image(post_copyback_path, post_copyback, quality=92)
        recorder.write_json(
            f"10_copyback_reassemble/{band_id}/band_crop_manifest.json",
            {
                "band_id": str(band_id),
                "source_page_number": int(source_page_number or 0),
                "y_top": int(getattr(band, "y_top", 0) or 0),
                "y_bottom": int(getattr(band, "y_bottom", 0) or 0),
                "coordinate_space": "band",
                "post_typeset": post_typeset_path,
                "post_copyback": post_copyback_path,
            },
        )
    except Exception:
        return


def _record_inpaint_skip_decision(
    *,
    band: Band,
    band_id: str,
    source_page_number: int | None,
    translated_page: dict | None,
    reason: str,
) -> None:
    try:
        from debug_tools import get_recorder
    except Exception:
        return
    recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    try:
        texts = [
            text
            for text in list((translated_page or {}).get("texts") or [])
            if isinstance(text, dict)
        ]
        skip_reasons: dict[str, int] = {}
        skipped_texts: list[dict] = []
        for index, text in enumerate(texts):
            if str(reason) != "unchanged_translation_skip":
                continue
            skip_reason = str(reason or "inpaint_skipped")
            skip_reasons[skip_reason] = int(skip_reasons.get(skip_reason, 0)) + 1
            text_id = _text_id_for(text, index)
            skipped_texts.append(
                {
                    "text_id": text_id,
                    "trace_id": str(text.get("trace_id") or _trace_id_for(text_id, band_id)),
                    "bbox": _coerce_bbox(text.get("bbox")),
                    "source_bbox": _coerce_bbox(text.get("source_bbox")),
                    "skip_reason": skip_reason,
                    "text": str(text.get("original") or text.get("text") or "")[:160],
                }
            )
        if not skip_reasons:
            skip_reasons[str(reason or "inpaint_skipped")] = int(len(texts))
        resolved_source_page_number = _source_page_number_from_page(translated_page, source_page_number) or 0
        payload = {
            "band_id": band_id,
            "source_page_number": int(resolved_source_page_number),
            "y_top": int(getattr(band, "y_top", 0) or 0),
            "y_bottom": int(getattr(band, "y_bottom", 0) or 0),
            "balloon_count": int(len(getattr(band, "balloons", []) or [])),
            "text_count": int(len(texts)),
            "skipped_text_count": int(len(skipped_texts)),
            "skipped": True,
            "inpaint_applied": False,
            "mask_applied": False,
            "copy_original": True,
            "reason": str(reason),
            "skip_reasons": skip_reasons,
            **_trace_metadata_payload(
                translated_page,
                band_id=band_id,
                source_page_number=source_page_number,
            ),
        }
        recorder.write_json(f"08_inpaint/{band_id}/inpaint_decision.json", payload)
        recorder.write_json(
            f"08_inpaint/{band_id}/skipped_texts.json",
            {
                "band_id": band_id,
                "reason": str(reason),
                "texts": skipped_texts,
            },
        )
        source_image = band.original_slice if band.original_slice is not None else band.strip_slice
        if source_image is not None:
            recorder.write_image(f"08_inpaint/{band_id}/00_band_original.jpg", source_image)
            recorder.write_image(f"08_inpaint/{band_id}/01_inpaint_skipped_original_copy.jpg", source_image)
    except Exception:
        return


@dataclass(frozen=True)
class BandStageOutput:
    stage_id: str
    _page: dict[str, Any] = field(repr=False)
    perf_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_page", copy.deepcopy(self._page))
        object.__setattr__(self, "perf_updates", MappingProxyType(dict(self.perf_updates)))

    def to_page_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._page)


@dataclass(frozen=True)
class BandImageStageOutput:
    stage_id: str
    _image: np.ndarray = field(repr=False)
    perf_updates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_image", np.array(self._image, copy=True))
        object.__setattr__(self, "perf_updates", MappingProxyType(copy.deepcopy(dict(self.perf_updates))))

    def to_image(self) -> np.ndarray:
        return np.array(self._image, copy=True)


def _band_to_page_dict(band: Band, page_idx: int, source_page_number: int | None = None) -> dict:
    """Converte uma Band para o formato dict que vision_stack.runtime aceita."""
    if band.strip_slice is None:
        raise ValueError("Band sem strip_slice; chame attach_band_slices primeiro")

    band_id = _band_id_for(source_page_number or page_idx + 1, page_idx)
    blocks = []
    height = int(band.strip_slice.shape[0])
    width = int(band.strip_slice.shape[1])
    for bubble_index, balloon in enumerate(band.balloons, start=1):
        bbox_local = [
            balloon.strip_bbox.x1,
            balloon.strip_bbox.y1 - band.y_top,
            balloon.strip_bbox.x2,
            balloon.strip_bbox.y2 - band.y_top,
        ]
        bubble_id = f"{band_id}_bubble_{bubble_index:03d}"
        bubble_inner_bbox = _inner_visual_rect_bbox(bbox_local, width=width, height=height)
        block = {
            "bbox": bbox_local,
            "confidence": balloon.confidence,
            "band_id": band_id,
            "bubble_id": bubble_id,
            "bubble_mask_bbox": list(bbox_local),
        }
        if bubble_inner_bbox is not None:
            block["bubble_inner_bbox"] = bubble_inner_bbox
        _attach_real_bubble_mask_to_block(block, band.strip_slice)
        blocks.append(block)

    page_number = int(source_page_number or page_idx + 1)
    band_index = int(page_idx + 1)

    return {
        "numero": page_number,
        "width": band.strip_slice.shape[1],
        "height": band.strip_slice.shape[0],
        "_vision_blocks": blocks,
        "_bubble_regions": [dict(block) for block in blocks],
        "_band_id": band_id,
        "_band_y_top": band.y_top,
        "_band_index": band_index,
        "_source_page_number": page_number,
    }


def _apply_copy_back_outside_balloons(
    band: Band,
    balloon_margin: int = 8,
    ocr_page: dict | None = None,
    rendered_slice: np.ndarray | None = None,
    cleaned_slice: np.ndarray | None = None,
) -> np.ndarray:
    """Copy-back defensivo: preserva pixels fora dos balões da banda.

    A máscara é a UNIÃO de:
      1. strip_bbox de cada balão (bbox do detector, em coords absolutas)
      2. balloon_bbox de cada texto no ocr_page (pode ser expandida por
         enrich_page_layout para cobrir a área branca real do balão)

    Sem a segunda fonte, texto renderizado na área expandida do balão seria
    sobrescrito pelo original, causando clipping visual nas bordas.
    """
    rendered = rendered_slice if rendered_slice is not None else band.rendered_slice
    if band.original_slice is None or rendered is None:
        raise ValueError("Band precisa de original_slice e rendered_slice")

    h, w = band.original_slice.shape[:2]
    mask_inside = np.zeros((h, w), dtype=bool)

    def _mark(x1: int, y1: int, x2: int, y2: int) -> None:
        bx1 = max(0, x1)
        by1 = max(0, y1)
        bx2 = min(w, x2)
        by2 = min(h, y2)
        if bx2 > bx1 and by2 > by1:
            mask_inside[by1:by2, bx1:bx2] = True

    # 1. Detector bbox (coords absolutas → band-local)
    for balloon in band.balloons:
        _mark(
            balloon.strip_bbox.x1 - balloon_margin,
            balloon.strip_bbox.y1 - band.y_top - balloon_margin,
            balloon.strip_bbox.x2 + balloon_margin,
            balloon.strip_bbox.y2 - band.y_top + balloon_margin,
        )

    # 2. balloon_bbox das camadas de texto (já em coords band-local)
    if ocr_page:
        def _bbox_area_local(bbox: list[int] | None) -> int:
            if bbox is None:
                return 0
            return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))

        def _as_band_local_bbox(value) -> list[int] | None:
            bbox = _coerce_bbox(value)
            if bbox is None:
                return None
            x1, y1, x2, y2 = bbox
            if y1 >= h and y2 > h:
                y1 -= int(band.y_top)
                y2 -= int(band.y_top)
            return [x1, y1, x2, y2]

        def _mark_bbox(value, margin: int) -> None:
            bbox = _as_band_local_bbox(value)
            if bbox is None:
                return
            bx1, by1, bx2, by2 = bbox
            _mark(
                bx1 - margin,
                by1 - margin,
                bx2 + margin,
                by2 + margin,
            )

        for txt in ocr_page.get("texts", []):
            if not isinstance(txt, dict):
                continue
            bbox = _as_band_local_bbox(txt.get("balloon_bbox") or txt.get("bbox"))
            if bbox is not None:
                _mark_bbox(bbox, balloon_margin)

            render_margin = max(2, min(6, int(balloon_margin)))
            _mark_bbox(txt.get("render_bbox"), render_margin)

            flags = {
                str(flag).strip().lower()
                for flag in (txt.get("qa_flags") or [])
                if str(flag).strip()
            }
            text_bbox = (
                _as_band_local_bbox(txt.get("text_pixel_bbox"))
                or _as_band_local_bbox(txt.get("source_bbox"))
                or _as_band_local_bbox(txt.get("bbox"))
            )
            collapsed_balloon = bool(
                bbox is not None
                and text_bbox is not None
                and _bbox_area_local(bbox) <= int(_bbox_area_local(text_bbox) * 1.25)
            )
            visual_contract = bool(
                collapsed_balloon
                or flags
                & {
                    "visual_text_only_inpaint_contract",
                    "dark_bubble_visual_glyph_mask_replaced_geometry",
                    "dark_bubble_visual_bbox_refined",
                    "dark_bubble_ellipse_bbox_mask",
                    "dark_bubble_connected_lobes_promoted",
                }
            )
            if visual_contract:
                _mark_bbox(txt.get("safe_text_box"), render_margin)
        inpaint_mask = _copy_back_inpaint_mask(band.original_slice, ocr_page)
        if inpaint_mask is not None and inpaint_mask.shape[:2] == mask_inside.shape:
            mask_inside |= inpaint_mask > 0
    if cleaned_slice is not None and cleaned_slice.shape == band.original_slice.shape:
        changed_by_inpaint = np.any(cleaned_slice != band.original_slice, axis=2).astype(np.uint8) * 255
        if np.any(changed_by_inpaint):
            changed_by_inpaint = cv2.dilate(
                changed_by_inpaint,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            mask_inside |= changed_by_inpaint > 0

    result = np.where(
        mask_inside[:, :, None],
        rendered,
        band.original_slice,
    )
    return result.astype(np.uint8)


def _copy_back_inpaint_mask(original_slice: np.ndarray, ocr_page: dict) -> np.ndarray | None:
    if original_slice is None or not isinstance(ocr_page, dict):
        return None
    engine_meta = ocr_page.get("_engine_preset") if isinstance(ocr_page.get("_engine_preset"), dict) else {}
    strategy = str((engine_meta or {}).get("mask_strategy") or "").strip().lower()
    if strategy not in {
        "segmentation_assisted",
        "roi_segmentation_assisted",
        "ocr_guided_segmentation",
        "ocr_guided_roi_segmentation",
    }:
        return None
    vision_blocks = [block for block in list(ocr_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    if not vision_blocks:
        return None
    try:
        from inpainter import _cjk_mask_kwargs_for_strip_page
        from vision_stack.runtime import vision_blocks_to_mask
    except Exception:
        return None
    try:
        return vision_blocks_to_mask(
            original_slice.shape,
            vision_blocks,
            image_rgb=original_slice,
            expand_mask=True,
            **_cjk_mask_kwargs_for_strip_page(ocr_page),
        )
    except Exception:
        return None


def _merge_translated_page_metadata(ocr_page: dict, translated_page: dict) -> dict:
    if not isinstance(translated_page, dict):
        return {"texts": []}

    merged_page = dict(translated_page)
    ocr_texts = list((ocr_page or {}).get("texts") or [])
    translated_texts = list((translated_page or {}).get("texts") or [])

    ocr_by_id = {
        text.get("id"): text
        for text in ocr_texts
        if isinstance(text, dict) and text.get("id")
    }

    merged_texts = []
    for index, translated_text in enumerate(translated_texts):
        if not isinstance(translated_text, dict):
            continue
        source_text = None
        text_id = translated_text.get("id")
        if text_id in ocr_by_id:
            source_text = ocr_by_id[text_id]
        elif index < len(ocr_texts) and isinstance(ocr_texts[index], dict):
            source_text = ocr_texts[index]
        merged_texts.append({**(source_text or {}), **translated_text})

    merged_page["texts"] = merged_texts

    if not merged_page.get("_vision_blocks"):
        merged_page["_vision_blocks"] = list((ocr_page or {}).get("_vision_blocks") or [])

    for key in (
        "numero",
        "width",
        "height",
        "_band_id",
        "_band_y_top",
        "_band_index",
        "_source_page_number",
        "page_profile",
        "engine_preset_id",
        "engine_preset",
        "_engine_preset",
        "_pipeline_artifacts",
        "_bubble_regions",
        "_negative_evidence",
    ):
        if (key not in merged_page or merged_page.get(key) in (None, "")) and key in ocr_page:
            merged_page[key] = copy.deepcopy(ocr_page[key])

    return merged_page


def _prepare_precomputed_ocr_page(precomputed_ocr_page: dict, page_dict: dict) -> dict:
    """Copia e completa uma página OCR já resolvida para o contrato band-local."""
    ocr_page = dict(precomputed_ocr_page or {})
    ocr_page["texts"] = [
        dict(text)
        for text in list(ocr_page.get("texts") or [])
        if isinstance(text, dict)
    ]
    ocr_page["_vision_blocks"] = [
        dict(block)
        for block in list(ocr_page.get("_vision_blocks") or [])
        if isinstance(block, dict)
    ]
    if isinstance(ocr_page.get("_ocr_stats"), dict):
        ocr_page["_ocr_stats"] = dict(ocr_page["_ocr_stats"])
    if isinstance(ocr_page.get("_negative_evidence"), dict):
        ocr_page["_negative_evidence"] = copy.deepcopy(ocr_page["_negative_evidence"])

    for key in (
        "numero",
        "width",
        "height",
        "_band_id",
        "_band_y_top",
        "_band_index",
        "_source_page_number",
    ):
        if key not in ocr_page and key in page_dict:
            ocr_page[key] = page_dict[key]
    return ocr_page


def _bbox_area(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_intersection_area(a: list[int] | None, b: list[int] | None) -> int:
    if a is None or b is None:
        return 0
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    inter = _bbox_intersection_area(a, b)
    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _bbox_overlap_min_ratio(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = max(1, min(_bbox_area(a), _bbox_area(b)))
    return _bbox_intersection_area(a, b) / float(denom)


def _bbox_center_distance(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return float("inf")
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _compact_text_for_similarity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _text_similarity(a: object, b: object) -> float:
    left = _compact_text_for_similarity(a)
    right = _compact_text_for_similarity(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    longer = max(len(left), len(right))
    shorter = min(len(left), len(right))
    if shorter >= 4 and (left in right or right in left):
        return shorter / float(longer)
    matches = sum(1 for char in left if char in right)
    return matches / float(longer)


def _should_replace_tight_bubble_bbox(
    text: dict,
    candidate_bbox: list[int] | None,
    *,
    current_key: str = "bubble_mask_bbox",
) -> bool:
    if not isinstance(text, dict) or candidate_bbox is None:
        return False
    current_bbox = _coerce_bbox(text.get(current_key))
    if current_bbox is None:
        return True
    current_area = _bbox_area(current_bbox)
    candidate_area = _bbox_area(candidate_bbox)
    if candidate_area <= max(0, current_area):
        return False
    text_bbox = _coerce_bbox(text.get("bbox") or text.get("text_pixel_bbox") or text.get("source_bbox"))
    text_area = _bbox_area(text_bbox)
    if text_area > 0:
        candidate_text_overlap = _bbox_intersection_area(text_bbox, candidate_bbox) / float(max(1, text_area))
        if candidate_text_overlap < 0.80:
            return False
    if current_area <= 0:
        return True
    return candidate_area >= max(current_area + 96, int(round(current_area * 1.35)))


def _should_replace_overlarge_derived_bubble_bbox(text: dict, candidate_bbox: list[int] | None) -> bool:
    if not isinstance(text, dict) or candidate_bbox is None:
        return False
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    if source not in WEAK_BUBBLE_MASK_SOURCES.union({"bbox_fallback"}):
        return False
    current_bbox = _coerce_bbox(text.get("bubble_mask_bbox"))
    if current_bbox is None:
        return False
    current_area = _bbox_area(current_bbox)
    candidate_area = _bbox_area(candidate_bbox)
    if current_area <= 0 or candidate_area <= 0:
        return False
    if current_area < max(candidate_area * 2.4, candidate_area + 2400):
        return False
    text_bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    if text_bbox is not None:
        text_area = max(1, _bbox_area(text_bbox))
        if _bbox_intersection_area(text_bbox, candidate_bbox) / float(text_area) < 0.80:
            return False
    return True


def _reject_unanchored_derived_bubble_contract(text: dict) -> None:
    if not isinstance(text, dict):
        return
    source = str(text.get("bubble_mask_source") or "").strip().lower()
    weak_bbox_sources = WEAK_BUBBLE_MASK_SOURCES
    if source not in weak_bbox_sources:
        return
    bubble_bbox = _coerce_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"))
    text_bbox = _coerce_bbox(
        text.get("_raw_text_evidence_bbox")
        or text.get("raw_text_evidence_bbox")
        or text.get("raw_text_bbox")
        or text.get("text_pixel_bbox")
        or text.get("layout_bbox")
        or text.get("bbox")
    )
    if bubble_bbox is None or text_bbox is None:
        return
    if _contract_value_present(text.get("bubble_mask")):
        pruned_mask, pruned_bbox, prune_error = _prune_mask_to_support_components(
            text.get("bubble_mask"),
            bubble_bbox,
            text_bbox,
        )
        if prune_error is not None:
            text.pop("bubble_mask", None)
            text.pop("bubble_inner_bbox", None)
            text["bubble_mask_source"] = "derived_white_crop_rejected"
            text["bubble_mask_error"] = prune_error
            return
        if pruned_mask is not None and pruned_bbox is not None and pruned_bbox != bubble_bbox:
            text["bubble_mask"] = pruned_mask
            text["bubble_mask_bbox"] = pruned_bbox
            bubble_bbox = pruned_bbox
    text_area = max(1, _bbox_area(text_bbox))
    bubble_area = max(1, _bbox_area(bubble_bbox))
    overlap_ratio = _bbox_intersection_area(text_bbox, bubble_bbox) / float(text_area)
    if overlap_ratio < 0.80:
        reason = "derived_mask_not_anchored_to_text"
    else:
        bubble_w = max(1, int(bubble_bbox[2]) - int(bubble_bbox[0]))
        bubble_h = max(1, int(bubble_bbox[3]) - int(bubble_bbox[1]))
        text_cx = (float(text_bbox[0]) + float(text_bbox[2])) / 2.0
        text_cy = (float(text_bbox[1]) + float(text_bbox[3])) / 2.0
        bubble_cx = (float(bubble_bbox[0]) + float(bubble_bbox[2])) / 2.0
        bubble_cy = (float(bubble_bbox[1]) + float(bubble_bbox[3])) / 2.0
        offset_x = abs(text_cx - bubble_cx) / float(bubble_w)
        offset_y = abs(text_cy - bubble_cy) / float(bubble_h)
        min_margin_x = min(max(0.0, text_cx - bubble_bbox[0]), max(0.0, bubble_bbox[2] - text_cx)) / float(bubble_w)
        min_margin_y = min(max(0.0, text_cy - bubble_bbox[1]), max(0.0, bubble_bbox[3] - text_cy)) / float(bubble_h)
        area_ratio = bubble_area / float(text_area)
        has_line_geometry = _contract_value_present(text.get("line_polygons"))
        unanchored_large = source in weak_bbox_sources and area_ratio >= 18.0 and (
            max(offset_x, offset_y) >= 0.28 or min(min_margin_x, min_margin_y) <= 0.08
        )
        unanchored_extreme = source in weak_bbox_sources and area_ratio >= 80.0
        flags = {
            str(flag).strip().lower()
            for flag in (text.get("qa_flags") or [])
            if str(flag).strip()
        }
        missing_glyph_evidence = bool(
            "raw_text_evidence_missing" in flags
            or "fast_fill_no_glyph_evidence" in flags
            or text.get("raw_text_evidence_missing")
        )
        suspect_large_text_region = (
            not has_line_geometry
            and source in weak_bbox_sources
            and text_area >= 50_000
            and missing_glyph_evidence
        )
        if not (unanchored_large or unanchored_extreme or suspect_large_text_region):
            return
        reason = "derived_mask_not_anchored_to_text"
    text.pop("bubble_mask", None)
    text.pop("bubble_inner_bbox", None)
    text["bubble_mask_source"] = "derived_white_crop_rejected"
    text["bubble_mask_error"] = reason


def _bbox_center_distance(a: list[int], b: list[int]) -> float:
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _bbox_fits_image(bbox: list[int], width: int, height: int, *, tolerance: int = 2) -> bool:
    return (
        bbox[0] >= -tolerance
        and bbox[1] >= -tolerance
        and bbox[2] <= width + tolerance
        and bbox[3] <= height + tolerance
    )


def _precomputed_bbox_reject_reason(
    label: str,
    value: Any,
    *,
    width: int,
    height: int,
    band_y_top: int,
) -> tuple[list[int] | None, str | None]:
    bbox = _coerce_bbox(value)
    if bbox is None:
        return None, f"{label}_missing_bbox"
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return bbox, f"{label}_invalid_bbox"
    if _bbox_area(bbox) <= 0:
        return bbox, f"{label}_invalid_bbox"
    if not _bbox_fits_image(bbox, width, height):
        if band_y_top:
            shifted = [bbox[0], bbox[1] - band_y_top, bbox[2], bbox[3] - band_y_top]
            if shifted[2] > shifted[0] and shifted[3] > shifted[1] and _bbox_fits_image(
                shifted,
                width,
                height,
            ):
                return bbox, f"{label}_mixed_coordinate_space"
        return bbox, f"{label}_out_of_bounds"
    return bbox, None


def _precomputed_text_has_value(text: dict) -> bool:
    raw = text.get("text", text.get("original", text.get("translated", "")))
    return bool(re.search(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", str(raw or "")))


def _validate_precomputed_ocr_page_geometry(ocr_page: dict, page_dict: dict, band: Band) -> str | None:
    texts = [text for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)]
    blocks = [block for block in list(ocr_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    stats = ocr_page.get("_ocr_stats") if isinstance(ocr_page.get("_ocr_stats"), dict) else {}

    try:
        width = int(ocr_page.get("width") or page_dict.get("width") or band.strip_slice.shape[1])
        height = int(ocr_page.get("height") or page_dict.get("height") or band.strip_slice.shape[0])
    except Exception:
        return "invalid_page_dimensions"
    if width <= 0 or height <= 0:
        return "invalid_page_dimensions"

    if not texts:
        if blocks or stats.get("macro_ocr_real") or int(stats.get("macro_ocr_block_count") or 0) > 0:
            return "empty_records"
        return "empty_page"

    if stats.get("macro_ocr_real") and not any(_precomputed_text_has_value(text) for text in texts):
        return "empty_records"

    try:
        band_y_top = int(getattr(band, "y_top", 0) or page_dict.get("_band_y_top") or 0)
    except Exception:
        band_y_top = 0

    for index, block in enumerate(blocks):
        _, reason = _precomputed_bbox_reject_reason(
            f"block_{index}",
            block.get("bbox"),
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        if reason:
            return reason

    for index, text in enumerate(texts):
        text_bbox, reason = _precomputed_bbox_reject_reason(
            f"text_{index}",
            text.get("bbox") or text.get("text_bbox") or text.get("text_pixel_bbox"),
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        if reason:
            return reason

        candidate_balloon = text.get("balloon_bbox")
        if candidate_balloon is None and len(blocks) == 1:
            candidate_balloon = blocks[0].get("bbox")
        if candidate_balloon is None:
            continue

        balloon_bbox, reason = _precomputed_bbox_reject_reason(
            f"text_{index}_balloon",
            candidate_balloon,
            width=width,
            height=height,
            band_y_top=band_y_top,
        )
        if reason:
            return reason

        text_area = _bbox_area(text_bbox)
        balloon_area = _bbox_area(balloon_bbox)
        intersection = _bbox_intersection_area(text_bbox, balloon_bbox)
        text_overlap = intersection / float(max(1, text_area))
        balloon_overlap = intersection / float(max(1, balloon_area))
        if text_overlap < 0.15 and balloon_overlap < 0.02:
            return "text_balloon_mismatch"

        balloon_width = max(1, balloon_bbox[2] - balloon_bbox[0])
        balloon_height = max(1, balloon_bbox[3] - balloon_bbox[1])
        max_reasonable_distance = max(balloon_width, balloon_height) * 1.35
        if _bbox_center_distance(text_bbox, balloon_bbox) > max_reasonable_distance:
            return "text_balloon_center_far"

    return None


def _run_band_ocr_stage(
    band: Band,
    *,
    runtime,
    page_dict: dict,
    precomputed_ocr_page: dict | None = None,
    work_title: str = "",
    work_title_user_provided: bool = False,
) -> BandStageOutput:
    def _call_runtime_ocr_stage() -> dict:
        if work_title or work_title_user_provided:
            try:
                return runtime.run_ocr_stage(
                    band.strip_slice,
                    page_dict,
                    work_title=work_title,
                    work_title_user_provided=work_title_user_provided,
                )
            except TypeError as exc:
                message = str(exc)
                if "unexpected keyword argument" not in message and "got an unexpected keyword" not in message:
                    raise
        return runtime.run_ocr_stage(band.strip_slice, page_dict)

    if isinstance(precomputed_ocr_page, dict):
        prepared_page = _prepare_precomputed_ocr_page(precomputed_ocr_page, page_dict)
        reject_reason = _validate_precomputed_ocr_page_geometry(prepared_page, page_dict, band)
        if reject_reason:
            fallback_page = _call_runtime_ocr_stage()
            if isinstance(fallback_page, dict):
                fallback_stats = dict(fallback_page.get("_ocr_stats") or {})
                fallback_stats.update(
                    {
                        "precomputed_ocr_rejected": True,
                        "precomputed_ocr_reject_reason": reject_reason,
                        "precomputed_ocr_runtime_fallback": True,
                    }
                )
                fallback_page["_ocr_stats"] = fallback_stats
            return BandStageOutput(
                "ocr",
                fallback_page,
                {
                    "ocr_precomputed_page": False,
                    "ocr_runtime_skipped": False,
                    "ocr_precomputed_page_rejected": True,
                    "ocr_precomputed_page_reject_reason": reject_reason,
                },
            )
        return BandStageOutput(
            "ocr",
            prepared_page,
            {
                "ocr_precomputed_page": True,
                "ocr_runtime_skipped": True,
            },
        )
    return BandStageOutput(
        "ocr",
        _call_runtime_ocr_stage(),
    )


def _candidate_crop_reocr_evidence_is_strong(evidence: dict) -> bool:
    try:
        significant_count = int(evidence.get("significant_component_count") or 0)
        significant_area = int(evidence.get("significant_area") or 0)
        light_count = int(evidence.get("inner_light_component_count") or 0)
        light_area = int(evidence.get("inner_light_area") or 0)
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
    except Exception:
        return False
    dark_text_on_light = (
        bool(evidence.get("has_inner_dark_text"))
        and significant_count >= 2
        and significant_area >= 300
        and bright_ratio >= 0.25
        and dark_ratio <= 0.12
    )
    light_text_on_dark = (
        (bool(evidence.get("has_inner_light_text")) or (light_count >= 1 and light_area >= 180))
        and light_count >= 1
        and light_area >= 180
        and dark_ratio >= 0.35
        and bright_ratio <= 0.16
    )
    return dark_text_on_light or light_text_on_dark


def _candidate_crop_reocr_allows_high_conf_dark_light_text(evidence: dict, *, confidence: float) -> bool:
    try:
        light_count = int(evidence.get("inner_light_component_count") or 0)
        light_area = int(evidence.get("inner_light_area") or 0)
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
    except Exception:
        return False
    return bool(
        float(confidence or 0.0) >= 0.86
        and (bool(evidence.get("has_inner_light_text")) or light_count >= 3)
        and light_area >= 900
        and dark_ratio >= 0.62
        and bright_ratio <= 0.22
    )


def _dark_bubble_evidence_supports_lobe_reocr(
    evidence: dict | None,
    *,
    min_dark_ratio: float,
    max_bright_ratio: float,
    min_light_area: int,
) -> bool:
    if not isinstance(evidence, dict):
        return False
    try:
        inner_light_area = int(evidence.get("inner_light_area") or 0)
        significant_area = int(evidence.get("significant_area") or 0)
        significant_count = int(evidence.get("significant_component_count") or 0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
    except Exception:
        return False
    has_light_text = (
        bool(evidence.get("has_inner_light_text"))
        or inner_light_area >= int(min_light_area)
        or (
            significant_count >= 1
            and significant_area >= max(90, int(min_light_area * 0.50))
            and bright_ratio >= 0.035
        )
    )
    return bool(
        has_light_text
        and dark_ratio >= float(min_dark_ratio)
        and bright_ratio <= float(max_bright_ratio)
    )


def _candidate_crop_reocr_evidence_allows_dark_panel_probe(evidence: dict, *, confidence: float) -> bool:
    try:
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
        inner_dark_area = int(evidence.get("inner_dark_area") or 0)
        inner_dark_count = int(evidence.get("inner_dark_component_count") or 0)
    except Exception:
        return False
    if float(confidence or 0.0) < 0.72:
        return False
    if dark_ratio < 0.45 or bright_ratio > 0.22:
        return False
    if not bool(evidence.get("has_inner_dark_text")) and inner_dark_count < 2:
        return False
    return inner_dark_area >= 18 or bright_ratio >= 0.045


def _candidate_crop_reocr_allows_sparse_dark_text(evidence: dict, *, confidence: float) -> bool:
    try:
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
        inner_dark_area = int(evidence.get("inner_dark_area") or 0)
        inner_dark_count = int(evidence.get("inner_dark_component_count") or 0)
    except Exception:
        return False
    if float(confidence or 0.0) < 0.72:
        return False
    if dark_ratio < 0.45 or bright_ratio > 0.24:
        return False
    has_internal_signal = (
        bool(evidence.get("has_inner_dark_text"))
        or inner_dark_area >= 18
        or inner_dark_count >= 2
        or bright_ratio >= 0.045
    )
    return bool(has_internal_signal)


def _shift_line_polygons_xy(value, delta_x: int, delta_y: int):
    if not isinstance(value, list):
        return value
    shifted = []
    for polygon in value:
        if not isinstance(polygon, list):
            shifted.append(polygon)
            continue
        shifted_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    shifted_polygon.append([int(round(float(point[0]))) + delta_x, int(round(float(point[1]))) + delta_y])
                    continue
                except Exception:
                    pass
            shifted_polygon.append(point)
        shifted.append(shifted_polygon)
    return shifted


def _candidate_crop_reocr_text_is_usable(text: dict) -> bool:
    raw = str(text.get("text") or text.get("original") or text.get("translated") or "").strip()
    compact = re.sub(r"[^A-Za-z0-9\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+", "", raw)
    if not compact:
        return False
    normalized = re.sub(r"[^A-Za-z0-9:/._#-]+", " ", raw).strip().lower()
    compact_latin = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if re.search(r"https?://|www\.|(?:d|)iscord\.gg|\.gg\b|\.co\b|\.com\b|\.net\b|\.org\b", normalized):
        return False
    if _candidate_crop_reocr_text_is_scanlation_credit(raw):
        return False
    if len(compact) <= 3 and any(ch.isdigit() for ch in compact):
        return False
    if len(compact) <= 3 and len(set(compact.lower())) <= 1:
        return False
    return True


def _candidate_crop_reocr_text_is_scanlation_credit(raw: object) -> bool:
    raw_text = str(raw or "").strip()
    if not raw_text:
        return False
    normalized = re.sub(r"[^A-Za-z0-9:/._#-]+", " ", raw_text).strip().lower()
    compact_latin = re.sub(r"[^a-z0-9]+", "", raw_text.lower())
    credit_terms = {
        "secret scans",
        "secretscans",
        "recruiting",
        "recrutando",
        "support us",
        "supportus",
        "join our discord",
        "our discord",
        "discord",
        "iscord.gg",
        "our team",
        "part of our team",
        "help us out",
        "join our",
        "patreon",
        "ko fi",
        "kofi",
        "read first",
        "readfirst",
        "jtl",
        "hiring jtls",
        "current and future projects",
        "future projects",
        "staff to help us release",
        "we are looking for",
        "translators",
        "proofreaders",
        "redrawers",
        "typesetters",
        "kr jp translators",
        "leave it blank",
        "mzfamily",
    }
    if any(term in normalized for term in credit_terms) or any(
        term in compact_latin
        for term in (
            "secretscans",
            "supportus",
            "joinourdiscord",
            "iscordgg",
            "partofourteam",
            "ourteam",
            "helpusout",
            "joinour",
            "discordgg",
            "patreon",
            "kofi",
            "readfirst",
            "leaveitblank",
            "recruiting",
            "recrutando",
            "hiringjtls",
            "currentandfutureprojects",
            "stafftohelpusrelease",
            "wearelookingfor",
            "translators",
            "proofreaders",
            "redrawers",
            "typesetters",
            "krjptranslators",
        )
    ):
        return True
    return False


def _candidate_crop_reocr_result_has_scanlation_credit(crop_page: dict | None) -> bool:
    if not isinstance(crop_page, dict):
        return False
    for text in list(crop_page.get("texts") or []):
        if not isinstance(text, dict):
            continue
        raw = text.get("text") or text.get("original") or text.get("raw_ocr") or text.get("translated") or ""
        if _candidate_crop_reocr_text_is_scanlation_credit(raw):
            return True
    return False


def _candidate_crop_reocr_result_scanlation_credit_texts(crop_page: dict | None) -> list[str]:
    if not isinstance(crop_page, dict):
        return []
    matched: list[str] = []
    for text in list(crop_page.get("texts") or []):
        if not isinstance(text, dict):
            continue
        raw = text.get("text") or text.get("original") or text.get("raw_ocr") or text.get("translated") or ""
        if _candidate_crop_reocr_text_is_scanlation_credit(raw):
            matched.append(str(raw or ""))
    return matched


def _candidate_crop_reocr_text_is_system_ui_status(text: dict) -> bool:
    raw = str(text.get("text") or text.get("original") or text.get("translated") or "").strip()
    if not raw:
        return False
    try:
        confidence = float(
            text.get("confidence_raw")
            if text.get("confidence_raw") not in (None, "")
            else text.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0
    if confidence < 0.72:
        return False
    normalized = re.sub(r"[^A-Za-z0-9']+", " ", raw).strip().lower()
    words = [word for word in normalized.split() if word]
    if len(words) < 3 or len(words) > 12:
        return False
    joined = " ".join(words)
    ui_terms = {
        "quest",
        "mission",
        "episode",
        "reward",
        "criteria",
        "level",
        "system",
        "synchronization",
        "sync",
        "host",
    }
    status_terms = {
        "begin",
        "begins",
        "complete",
        "completed",
        "shown",
        "shortly",
        "start",
        "started",
        "starts",
        "establish",
        "established",
        "reward",
    }
    if any(term in words for term in ui_terms) and any(term in words for term in status_terms):
        return True
    return any(
        phrase in joined
        for phrase in (
            "main quest",
            "quest introduction",
            "quest completion",
            "the episode starts",
            "quest reward",
            "search complete",
        )
    )


def _candidate_crop_reocr_text_is_dark_bubble_dialogue(text: dict) -> bool:
    raw = str(text.get("text") or text.get("original") or text.get("translated") or "").strip()
    if not raw:
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    profiles = {
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("layout_profile") or "").strip().lower(),
    }
    if source != "image_dark_bubble_mask" and "dark_bubble" not in profiles:
        return False
    try:
        confidence = float(
            text.get("confidence_raw")
            if text.get("confidence_raw") not in (None, "")
            else text.get("confidence", 0.0)
        )
    except Exception:
        confidence = 0.0
    if confidence < 0.72:
        return False
    if re.search(r"https?://|www\.|discord\.gg|\.co\b|\.com\b|\.net\b|\.org\b", raw.lower()):
        return False
    normalized = re.sub(r"[^A-Za-z0-9']+", " ", raw).strip()
    words = [word for word in normalized.split() if word]
    if len(words) < 2 or len(words) > 26:
        return False
    alpha = sum(1 for ch in normalized if ch.isalpha())
    return alpha >= 8


def _candidate_crop_reocr_bbox_is_reasonable(text: dict, *, crop_width: int, crop_height: int) -> bool:
    bbox = _coerce_bbox(text.get("bbox") or text.get("text_pixel_bbox"))
    if bbox is None:
        return True
    bbox_w = max(0, bbox[2] - bbox[0])
    bbox_h = max(0, bbox[3] - bbox[1])
    if bbox_w <= 0 or bbox_h <= 0:
        return False
    crop_area = max(1, int(crop_width) * int(crop_height))
    bbox_area = bbox_w * bbox_h
    if bbox_area > int(crop_area * 0.35):
        return _candidate_crop_reocr_text_is_system_ui_status(text) or _candidate_crop_reocr_text_is_dark_bubble_dialogue(text)
    source = str(text.get("ocr_source") or "").strip().lower()
    if (
        not source.startswith("candidate_crop_direct_paddle")
        and bbox_w > int(crop_width * 0.86)
        and bbox_h > int(crop_height * 0.28)
    ):
        return _candidate_crop_reocr_text_is_system_ui_status(text) or _candidate_crop_reocr_text_is_dark_bubble_dialogue(text)
    return True


def _candidate_crop_reocr_bbox_fits_balloon(text: dict, balloon_bbox: list[int]) -> bool:
    bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    balloon = _coerce_bbox(balloon_bbox)
    if bbox is None or balloon is None:
        return True
    bbox_area = _bbox_area(bbox)
    balloon_area = _bbox_area(balloon)
    if bbox_area <= 0 or balloon_area <= 0:
        return False
    overlap = _bbox_intersection_area(bbox, balloon)
    if overlap / float(bbox_area) < 0.88:
        return False
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    lx1, ly1, lx2, ly2 = [int(v) for v in balloon]
    bbox_w = max(1, bx2 - bx1)
    bbox_h = max(1, by2 - by1)
    balloon_w = max(1, lx2 - lx1)
    balloon_h = max(1, ly2 - ly1)
    if bbox_w > int(balloon_w * 0.86) and bbox_h > int(balloon_h * 0.70):
        return False
    touches_right_edge = bx2 >= lx2 - 2
    touches_left_edge = bx1 <= lx1 + 2
    raw_text = str(text.get("text") or text.get("raw_ocr") or text.get("original") or "").strip()
    word_count = len(re.findall(r"[A-Za-z0-9']+", raw_text))
    if (
        word_count >= 7
        and bbox_w > int(balloon_w * 0.55)
        and (touches_right_edge or touches_left_edge)
    ):
        return False
    return True


def _filter_dark_bubble_reocr_to_balloon(
    texts: list[dict],
    blocks: list[dict],
    *,
    bubble_bbox: list[int],
) -> tuple[list[dict], list[dict]]:
    kept_texts: list[dict] = []
    kept_indices: set[int] = set()
    for index, text in enumerate(texts):
        if not isinstance(text, dict):
            continue
        if _candidate_crop_reocr_bbox_fits_balloon(text, bubble_bbox):
            kept_texts.append(text)
            kept_indices.add(index)
            continue
        flags = list(text.get("qa_flags") or [])
        if "dark_bubble_reocr_cross_lobe_rejected" not in flags:
            flags.append("dark_bubble_reocr_cross_lobe_rejected")
        text["qa_flags"] = flags
    if len(kept_texts) == len(texts):
        return texts, blocks
    if len(blocks) == len(texts):
        kept_blocks = [
            block
            for index, block in enumerate(blocks)
            if index in kept_indices and isinstance(block, dict)
        ]
        return kept_texts, kept_blocks
    kept_blocks: list[dict] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _candidate_crop_reocr_bbox_fits_balloon(block, bubble_bbox):
            kept_blocks.append(block)
    return kept_texts, kept_blocks


def _map_crop_ocr_page_to_band(
    crop_page: dict,
    *,
    band_page: dict,
    band_id: str,
    balloon_local_bbox: list[int],
    crop_left: int,
    crop_top: int,
    candidate_index: int,
) -> tuple[list[dict], list[dict]]:
    mapped_texts: list[dict] = []
    crop_width = int(crop_page.get("width") or 0)
    crop_height = int(crop_page.get("height") or 0)
    for index, text in enumerate(list(crop_page.get("texts") or [])):
        if not isinstance(text, dict) or not _precomputed_text_has_value(text):
            continue
        if not _candidate_crop_reocr_text_is_usable(text):
            continue
        if crop_width > 0 and crop_height > 0 and not _candidate_crop_reocr_bbox_is_reasonable(
            text,
            crop_width=crop_width,
            crop_height=crop_height,
        ):
            continue
        mapped = copy.deepcopy(text)
        text_id = str(mapped.get("text_id") or mapped.get("id") or f"reocr_{candidate_index:03d}_{index + 1:03d}")
        mapped["id"] = text_id
        mapped["text_id"] = text_id
        for key in (
            "bbox",
            "source_bbox",
            "text_pixel_bbox",
            "balloon_bbox",
            "layout_bbox",
            "bubble_mask_bbox",
            "bubble_inner_bbox",
        ):
            shifted_bbox = _shift_bbox_xy(mapped.get(key), crop_left, crop_top)
            if shifted_bbox is not None:
                mapped[key] = shifted_bbox
        if not _coerce_bbox(mapped.get("balloon_bbox")):
            mapped["balloon_bbox"] = list(balloon_local_bbox)
        if not _coerce_bbox(mapped.get("bubble_mask_bbox")):
            mapped["bubble_mask_bbox"] = list(balloon_local_bbox)
        if "line_polygons" in mapped:
            mapped["line_polygons"] = _shift_line_polygons_xy(mapped.get("line_polygons"), crop_left, crop_top)
        mapped["ocr_source"] = "candidate_crop_reocr"
        mapped["reocr_candidate_index"] = int(candidate_index)
        mapped["reocr_crop_offset"] = [int(crop_left), int(crop_top)]
        mapped["band_id"] = band_id
        mapped["_band_id"] = band_id
        mapped_texts.append(mapped)

    mapped_blocks: list[dict] = []
    for block in list(crop_page.get("_vision_blocks") or []):
        if not isinstance(block, dict):
            continue
        mapped_block = copy.deepcopy(block)
        shifted_bbox = _shift_bbox_xy(mapped_block.get("bbox"), crop_left, crop_top)
        if shifted_bbox is not None:
            mapped_block["bbox"] = shifted_bbox
        else:
            mapped_block["bbox"] = list(balloon_local_bbox)
        mapped_block["band_id"] = band_id
        mapped_block["reocr_candidate_index"] = int(candidate_index)
        mapped_block["ocr_source"] = "candidate_crop_reocr"
        mapped_blocks.append(mapped_block)
    if not mapped_blocks and mapped_texts:
        mapped_blocks.append(
            {
                "bbox": list(balloon_local_bbox),
                "confidence": 0.0,
                "band_id": band_id,
                "source": "candidate_crop_reocr_balloon",
            }
        )
    return mapped_texts, mapped_blocks


def _paddle_runs_to_crop_page(
    runs: list[dict],
    *,
    crop_width: int,
    crop_height: int,
    scale: float,
    source: str,
) -> dict:
    lines: list[dict] = []
    inv_scale = 1.0 / float(scale or 1.0)
    for run in runs:
        text = str(run.get("text") or "").strip()
        if not text:
            continue
        pts = run.get("bbox_pts")
        if not isinstance(pts, (list, tuple)) or len(pts) < 4:
            continue
        polygon: list[list[int]] = []
        for point in pts:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                polygon.append(
                    [
                        max(0, min(crop_width, int(round(float(point[0]) * inv_scale)))),
                        max(0, min(crop_height, int(round(float(point[1]) * inv_scale)))),
                    ]
                )
            except Exception:
                continue
        if len(polygon) < 4:
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        lines.append(
            {
                "text": text,
                "bbox": bbox,
                "polygon": polygon,
                "confidence": float(run.get("confidence", 0.0) or 0.0),
            }
        )
    if not lines:
        return {"texts": [], "_vision_blocks": [], "width": crop_width, "height": crop_height}
    lines.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    bbox = [
        min(int(item["bbox"][0]) for item in lines),
        min(int(item["bbox"][1]) for item in lines),
        max(int(item["bbox"][2]) for item in lines),
        max(int(item["bbox"][3]) for item in lines),
    ]
    text = " ".join(str(item["text"]).strip() for item in lines if str(item["text"]).strip()).strip()
    confidence = float(np.mean([float(item["confidence"]) for item in lines])) if lines else 0.0
    record = {
        "id": "direct_paddle_reocr_001",
        "text_id": "direct_paddle_reocr_001",
        "text": text,
        "raw_ocr": text,
        "original": text,
        "normalized_ocr": text,
        "normalized_text_final": text,
        "bbox": list(bbox),
        "source_bbox": list(bbox),
        "text_pixel_bbox": list(bbox),
        "line_polygons": [item["polygon"] for item in lines],
        "confidence": confidence,
        "confidence_raw": confidence,
        "ocr_confidence": confidence,
        "ocr_source": source,
        "qa_flags": ["candidate_crop_direct_paddle_reocr"],
    }
    return {
        "texts": [record],
        "_vision_blocks": [
            {
                "bbox": list(bbox),
                "confidence": confidence,
                "source": source,
            }
        ],
        "width": crop_width,
        "height": crop_height,
    }


def _run_direct_paddle_candidate_crop_reocr(crop: np.ndarray, *, idioma_origem: str = "en") -> dict:
    try:
        from ocr_legacy.recognizer_paddle import is_paddle_available, run_paddle_primary_recognition
        from vision_stack.ocr import normalize_paddleocr_language
    except Exception:
        return {"texts": [], "_vision_blocks": []}
    if not is_paddle_available():
        return {"texts": [], "_vision_blocks": []}
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return {"texts": [], "_vision_blocks": []}
    crop_height, crop_width = crop.shape[:2]
    variants: list[tuple[str, np.ndarray, float]] = []
    native = crop.astype(np.uint8)
    swapped = cv2.cvtColor(native, cv2.COLOR_RGB2BGR)
    variants.append(("candidate_crop_direct_paddle_native", native, 1.0))
    variants.append((
        "candidate_crop_direct_paddle_native_x2",
        cv2.resize(native, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        2.0,
    ))
    variants.append(("candidate_crop_direct_paddle", swapped, 1.0))
    variants.append((
        "candidate_crop_direct_paddle_x2",
        cv2.resize(swapped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        2.0,
    ))
    variants.append(("candidate_crop_direct_paddle_native_inverted", cv2.bitwise_not(native), 1.0))
    variants.append(("candidate_crop_direct_paddle_inverted", cv2.bitwise_not(swapped), 1.0))
    best_page = {"texts": [], "_vision_blocks": [], "width": crop_width, "height": crop_height}
    best_score = 0.0
    lang = normalize_paddleocr_language(idioma_origem)
    for source, image_bgr, scale in variants:
        try:
            runs = run_paddle_primary_recognition(image_bgr, use_gpu=True, lang=lang)
        except Exception:
            continue
        page = _paddle_runs_to_crop_page(
            runs,
            crop_width=crop_width,
            crop_height=crop_height,
            scale=scale,
            source=source,
        )
        texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
        if not texts:
            continue
        text_value = " ".join(str(text.get("text") or "").strip() for text in texts)
        score = len(text_value) + 50.0 * float(texts[0].get("confidence", 0.0) or 0.0)
        if score > best_score:
            best_score = score
            best_page = page
    return best_page


def _detect_dark_rect_panel_frame_bbox(image: np.ndarray, bbox: BBox) -> list[int] | None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if w < 120 or h < 60:
        return None
    inner = gray[max(4, h // 8) : max(5, h - max(4, h // 8)), max(4, w // 8) : max(5, w - max(4, w // 8))]
    if inner.size == 0 or float(np.median(inner)) > 70.0:
        return None
    edge = gray >= 120
    row_coverage = np.mean(edge, axis=1)
    col_coverage = np.mean(edge, axis=0)
    horizontal_rows = np.where(row_coverage >= 0.58)[0]
    vertical_cols = np.where(col_coverage >= 0.18)[0]
    if horizontal_rows.size < 2 or vertical_cols.size < 2:
        return None
    top_candidates = horizontal_rows[horizontal_rows < int(h * 0.38)]
    bottom_candidates = horizontal_rows[horizontal_rows > int(h * 0.62)]
    left_candidates = vertical_cols[vertical_cols < int(w * 0.38)]
    right_candidates = vertical_cols[vertical_cols > int(w * 0.62)]
    if not (top_candidates.size and bottom_candidates.size and left_candidates.size and right_candidates.size):
        return None
    top = int(top_candidates[0])
    bottom = int(bottom_candidates[-1])
    left = int(left_candidates[0])
    right = int(right_candidates[-1])
    if right - left < int(w * 0.38) or bottom - top < int(h * 0.22):
        return None
    panel = gray[top : bottom + 1, left : right + 1]
    if panel.size == 0:
        return None
    if float(np.median(panel)) > 82.0:
        return None
    return [x1 + left, y1 + top, x1 + right + 1, y1 + bottom + 1]


def _detect_dark_oval_bubble_bbox(image: np.ndarray, bbox: BBox) -> list[int] | None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(bbox.x1)))
    x2 = max(0, min(width, int(bbox.x2)))
    y1 = max(0, min(height, int(bbox.y1)))
    y2 = max(0, min(height, int(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    if w < 90 or h < 60:
        return None
    inner = gray[max(4, h // 5) : max(5, h - max(4, h // 5)), max(4, w // 5) : max(5, w - max(4, w // 5))]
    if inner.size == 0 or float(np.median(inner)) > 58.0:
        return None
    rgb = crop.astype(np.uint8)
    blue_glow = (
        (rgb[:, :, 2].astype(np.int16) >= rgb[:, :, 0].astype(np.int16) + 18)
        & (rgb[:, :, 1].astype(np.int16) >= rgb[:, :, 0].astype(np.int16) + 8)
        & (gray >= 18)
    ).astype(np.uint8) * 255
    edge = cv2.bitwise_or(cv2.Canny(gray, 24, 90), cv2.Canny(255 - gray, 24, 90))
    border_signal = cv2.bitwise_or(edge, blue_glow)
    border_signal = cv2.morphologyEx(
        border_signal,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    row_coverage = np.mean(border_signal > 0, axis=1)
    col_coverage = np.mean(border_signal > 0, axis=0)
    top_rows = np.where(row_coverage[: max(1, int(h * 0.35))] >= 0.45)[0]
    bottom_rows = np.where(row_coverage[int(h * 0.65) :] >= 0.45)[0]
    left_cols = np.where(col_coverage[: max(1, int(w * 0.35))] >= 0.12)[0]
    right_cols = np.where(col_coverage[int(w * 0.65) :] >= 0.12)[0]
    if top_rows.size and bottom_rows.size and left_cols.size and right_cols.size:
        return None
    contours, _ = cv2.findContours(border_signal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(2800, int(w * h * 0.20))
    best: tuple[int, list[int]] | None = None
    for contour in contours:
        if len(contour) < 12:
            continue
        cx, cy, cw, ch = [int(v) for v in cv2.boundingRect(contour)]
        area = cw * ch
        if area < min_area or area > int(w * h * 0.98):
            continue
        aspect = cw / float(max(1, ch))
        if aspect < 0.85 or aspect > 3.8:
            continue
        component = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, 2)
        signal_pixels = int(np.count_nonzero(component & border_signal))
        if signal_pixels < max(90, int((cw + ch) * 0.36)):
            continue
        candidate = [x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch]
        if best is None or signal_pixels > best[0]:
            best = (signal_pixels, candidate)
    if best is not None:
        return best[1]
    # Detector boxes for dark bubbles are often already tight; accept them only
    # when an ellipse-shaped cyan edge exists around a dark interior.
    center = (w // 2, h // 2)
    axes = (max(2, int(w * 0.44)), max(2, int(h * 0.42)))
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ring, center, axes, 0, 0, 360, 255, 3)
    ring_pixels = int(np.count_nonzero(ring))
    ring_signal = int(np.count_nonzero((border_signal > 0) & (ring > 0)))
    if ring_pixels > 0 and ring_signal / float(ring_pixels) >= 0.045:
        return [x1, y1, x2, y2]
    return None


def _expanded_candidate_crop_bbox(
    *,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    pad_x = 20
    pad_y = max(40, int(round((y2 - y1) * 0.7)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _dark_panel_probe_candidate_crop_bbox(
    *,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    pad_x = max(20, int(round((x2 - x1) * 0.18)))
    pad_y = max(70, int(round((y2 - y1) * 1.3)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _partial_dark_bubble_probe_crop_bbox(
    *,
    width: int,
    height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    pad_x = max(60, int(round(box_w * 0.55)))
    pad_top = max(120, int(round(box_h * 3.0)))
    pad_bottom = max(70, int(round(box_h * 1.9)))
    return (
        max(0, int(x1) - pad_x),
        max(0, int(y1) - pad_top),
        min(width, int(x2) + pad_x),
        min(height, int(y2) + pad_bottom),
    )


def _candidate_has_dark_rect_panel_frame(image: np.ndarray, bbox: BBox) -> bool:
    return _detect_dark_rect_panel_frame_bbox(image, bbox) is not None


def _annotate_dark_bubble_recovery(
    texts: list[dict],
    blocks: list[dict],
    *,
    bubble_bbox: list[int],
) -> None:
    for item in list(texts) + list(blocks):
        if not isinstance(item, dict):
            continue
        item["bubble_mask_source"] = "image_dark_bubble_mask"
        item["bubble_mask_bbox"] = list(bubble_bbox)
        item["balloon_bbox"] = list(bubble_bbox)
        item["layout_profile"] = item.get("layout_profile") or "dark_bubble"
        item["block_profile"] = item.get("block_profile") or "dark_bubble"
        flags = list(item.get("qa_flags") or [])
        if "candidate_crop_direct_paddle_reocr" not in flags:
            flags.append("candidate_crop_direct_paddle_reocr")
        if "dark_bubble_oval_reocr" not in flags:
            flags.append("dark_bubble_oval_reocr")
        item["qa_flags"] = flags
        if item in texts:
            _strip_false_trailing_dark_bubble_fragment(item)
        item.pop("bubble_mask_error", None)
        item.pop("bubbleMaskError", None)


def _negative_contrast_ocr_crop(crop: np.ndarray) -> np.ndarray | None:
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return None
    source = crop.astype(np.uint8)
    if source.ndim == 2:
        gray = source
    elif source.ndim == 3 and source.shape[2] >= 3:
        gray = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        return None
    inverted = cv2.bitwise_not(gray)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(inverted)
    except Exception:
        contrasted = inverted
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)


def _dark_bubble_bbox_is_only_text_candidate(detected_bbox: list[int] | None, candidate_bbox: list[int]) -> bool:
    detected = _coerce_bbox(detected_bbox)
    candidate = _coerce_bbox(candidate_bbox)
    if detected is None or candidate is None:
        return False
    detected_area = _bbox_area(detected)
    candidate_area = _bbox_area(candidate)
    if detected_area <= 0 or candidate_area <= 0:
        return False
    dx1, dy1, dx2, dy2 = [int(v) for v in detected]
    cx1, cy1, cx2, cy2 = [int(v) for v in candidate]
    detected_w = max(1, dx2 - dx1)
    detected_h = max(1, dy2 - dy1)
    candidate_w = max(1, cx2 - cx1)
    candidate_h = max(1, cy2 - cy1)
    overlap = _bbox_intersection_area(detected, candidate)
    overlap_ratio = overlap / float(max(1, min(detected_area, candidate_area)))
    if overlap_ratio < 0.82:
        return False
    if detected_area < int(candidate_area * 2.2):
        return True
    if detected_w < int(candidate_w * 1.25) or detected_h < int(candidate_h * 1.35):
        return True
    return False


def _candidate_reocr_support_mask(shape: tuple[int, int], texts: list[dict]) -> np.ndarray | None:
    height, width = shape
    if height <= 0 or width <= 0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for text in texts:
        if not isinstance(text, dict):
            continue
        polygons = text.get("line_polygons")
        if isinstance(polygons, list):
            for polygon in polygons:
                pts: list[list[int]] = []
                if not isinstance(polygon, list):
                    continue
                for point in polygon:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            pts.append([int(round(float(point[0]))), int(round(float(point[1])))])
                        except (TypeError, ValueError):
                            continue
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 255)
        bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
        if bbox:
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    return mask if np.any(mask) else None


def _tight_light_text_crop_bbox_for_dark_lobe(image: np.ndarray, bbox: list[int]) -> list[int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return [x1, y1, x2, y2]
    gray = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    light = (gray >= 150).astype(np.uint8) * 255
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((light > 0).astype(np.uint8), 8)
    keep = np.zeros_like(light)
    for idx in range(1, count):
        cx, cy, cw, ch, area = [int(v) for v in stats[idx]]
        if area < 8 or cw < 2 or ch < 4:
            continue
        if cw > int((x2 - x1) * 0.92) and ch < 8:
            continue
        keep[labels == idx] = 255
    if not np.any(keep):
        return [x1, y1, x2, y2]
    ys, xs = np.where(keep > 0)
    tx1 = int(xs.min()) + x1
    ty1 = int(ys.min()) + y1
    tx2 = int(xs.max()) + 1 + x1
    ty2 = int(ys.max()) + 1 + y1
    text_w = max(1, tx2 - tx1)
    text_h = max(1, ty2 - ty1)
    pad_x = max(28, int(round(text_w * 0.18)))
    pad_y = max(28, int(round(text_h * 0.28)))
    return [
        max(x1, tx1 - pad_x),
        max(y1, ty1 - pad_y),
        min(x2, tx2 + pad_x),
        min(y2, ty2 + pad_y),
    ]


def _annotate_dark_panel_recovery(
    texts: list[dict],
    blocks: list[dict],
    *,
    image: np.ndarray,
    panel_bbox: list[int],
) -> None:
    color_metrics: dict = {}
    try:
        from inpainter.mask_builder import _sample_dark_panel_effect_colors

        support_mask = _candidate_reocr_support_mask(image.shape[:2], texts)
        color_metrics = _sample_dark_panel_effect_colors(image.astype(np.uint8), panel_bbox, support_mask)
    except Exception:
        color_metrics = {}
    for item in list(texts) + list(blocks):
        if not isinstance(item, dict):
            continue
        item["bubble_mask_source"] = "image_dark_panel_mask"
        item["bubble_mask_bbox"] = list(panel_bbox)
        item["balloon_bbox"] = list(panel_bbox)
        item["card_panel_text_context"] = True
        item["layout_profile"] = item.get("layout_profile") or "dark_panel"
        item["block_profile"] = item.get("block_profile") or "dark_panel"
        if color_metrics:
            item["dark_panel_effect_colors"] = copy.deepcopy(color_metrics)
            qa_metrics = item.setdefault("qa_metrics", {})
            if isinstance(qa_metrics, dict):
                qa_metrics["image_dark_panel_mask"] = {
                    "source": "image_dark_panel_mask",
                    "detection_space": "candidate_crop_rect_frame",
                    "mask_bbox": list(panel_bbox),
                    **copy.deepcopy(color_metrics),
                }
        flags = list(item.get("qa_flags") or [])
        for flag in ("bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected", "rejected_derived_bubble_mask"):
            while flag in flags:
                flags.remove(flag)
        if "candidate_crop_direct_paddle_reocr" in flags:
            item["qa_flags"] = flags
        elif flags:
            item["qa_flags"] = flags
        item.pop("bubble_mask_error", None)
        item.pop("bubbleMaskError", None)


def _recover_empty_ocr_with_candidate_crops(
    band: Band,
    *,
    runtime,
    page_dict: dict,
    band_id: str,
    work_title: str = "",
    work_title_user_provided: bool = False,
) -> BandStageOutput:
    try:
        from strip.detect_balloons import _inner_dark_text_evidence
    except Exception:
        _inner_dark_text_evidence = None

    image = band.strip_slice
    height, width = image.shape[:2]
    recovered_texts: list[dict] = []
    recovered_blocks: list[dict] = []
    rejected_scanlation_credit_texts: list[str] = []
    attempts = 0
    candidate_count = 0

    def _call_runtime(crop: np.ndarray, crop_page: dict) -> dict:
        if work_title or work_title_user_provided:
            try:
                return runtime.run_ocr_stage(
                    crop,
                    crop_page,
                    work_title=work_title,
                    work_title_user_provided=work_title_user_provided,
                )
            except TypeError as exc:
                message = str(exc)
                if "unexpected keyword argument" not in message and "got an unexpected keyword" not in message:
                    raise
        return runtime.run_ocr_stage(crop, crop_page)

    for candidate_index, balloon in enumerate(list(band.balloons or [])):
        local_bbox = [
            int(balloon.strip_bbox.x1),
            int(balloon.strip_bbox.y1) - int(band.y_top),
            int(balloon.strip_bbox.x2),
            int(balloon.strip_bbox.y2) - int(band.y_top),
        ]
        x1, y1, x2, y2 = local_bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        candidate_count += 1
        confidence = float(getattr(balloon, "confidence", 0.0) or 0.0)
        crop_left, crop_top, crop_right, crop_bottom = _expanded_candidate_crop_bbox(
            width=width,
            height=height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if crop_right <= crop_left or crop_bottom <= crop_top:
            continue
        rect_panel_bbox = None
        dark_oval_bbox = None
        probe_left, probe_top, probe_right, probe_bottom = _dark_panel_probe_candidate_crop_bbox(
            width=width,
            height=height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if confidence >= 0.72:
            rect_panel_bbox = _detect_dark_rect_panel_frame_bbox(
                image,
                BBox(probe_left, probe_top, probe_right, probe_bottom),
            )
        if confidence >= 0.35:
            dark_oval_bbox = _detect_dark_oval_bubble_bbox(
                image,
                BBox(probe_left, probe_top, probe_right, probe_bottom),
            )
        allow_dark_panel_probe = False
        allow_dark_bubble_probe = False
        allow_sparse_dark_bubble_reocr = False
        high_conf_dark_light_text = False
        if _inner_dark_text_evidence is not None:
            evidence = _inner_dark_text_evidence(image, BBox(x1, y1, x2, y2))
            evidence_is_strong = _candidate_crop_reocr_evidence_is_strong(evidence)
            high_conf_dark_light_text = _candidate_crop_reocr_allows_high_conf_dark_light_text(
                evidence,
                confidence=confidence,
            )
            allow_dark_panel_probe = (
                rect_panel_bbox is not None
                and _candidate_crop_reocr_evidence_allows_dark_panel_probe(
                    evidence,
                    confidence=confidence,
                )
            )
            allow_dark_bubble_probe = (
                dark_oval_bbox is not None
                and _dark_bubble_evidence_supports_lobe_reocr(
                    evidence,
                    min_dark_ratio=0.35,
                    max_bright_ratio=0.24,
                    min_light_area=180,
                )
            )
            allow_sparse_dark_bubble_reocr = (
                dark_oval_bbox is not None
                and _candidate_crop_reocr_allows_sparse_dark_text(evidence, confidence=confidence)
            )
            if (
                not evidence_is_strong
                and not allow_dark_panel_probe
                and not allow_dark_bubble_probe
                and not allow_sparse_dark_bubble_reocr
                and not high_conf_dark_light_text
            ):
                continue
            if (
                allow_dark_panel_probe
                or allow_dark_bubble_probe
                or allow_sparse_dark_bubble_reocr
                or high_conf_dark_light_text
            ) and not evidence_is_strong:
                crop_left, crop_top, crop_right, crop_bottom = (
                    probe_left,
                    probe_top,
                    probe_right,
                    probe_bottom,
                )
        else:
            evidence = {}
        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        crop_block_bbox = [0, 0, int(crop.shape[1]), int(crop.shape[0])]
        crop_balloon_bbox = [
            int(x1 - crop_left),
            int(y1 - crop_top),
            int(x2 - crop_left),
            int(y2 - crop_top),
        ]
        crop_page = dict(page_dict)
        crop_page.update(
            {
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "_band_id": band_id,
                "_candidate_crop_reocr": True,
                "_candidate_crop_offset": [int(crop_left), int(crop_top)],
                "_vision_blocks": [
                    {
                        "bbox": crop_block_bbox,
                        "confidence": float(getattr(balloon, "confidence", 0.0) or 0.0),
                        "band_id": band_id,
                        "bubble_id": f"{band_id}_reocr_{candidate_index:03d}",
                        "balloon_bbox": crop_balloon_bbox,
                        "bubble_mask_bbox": crop_balloon_bbox,
                    }
                ],
                "_bubble_regions": [],
            }
        )
        attempts += 1
        crop_result = _call_runtime(crop, crop_page)
        if not isinstance(crop_result, dict):
            continue
        crop_scanlation_texts = _candidate_crop_reocr_result_scanlation_credit_texts(crop_result)
        if crop_scanlation_texts:
            rejected_scanlation_credit_texts.extend(crop_scanlation_texts)
            continue
        texts, blocks = _map_crop_ocr_page_to_band(
            crop_result,
            band_page=page_dict,
            band_id=band_id,
            balloon_local_bbox=[x1, y1, x2, y2],
            crop_left=crop_left,
            crop_top=crop_top,
            candidate_index=candidate_index,
        )
        has_rect_panel_frame = rect_panel_bbox is not None
        has_dark_oval_bubble = dark_oval_bbox is not None
        if (
            _dark_bubble_evidence_supports_lobe_reocr(
                evidence,
                min_dark_ratio=0.35,
                max_bright_ratio=0.28,
                min_light_area=180,
            )
            or allow_dark_panel_probe
            or allow_dark_bubble_probe
            or allow_sparse_dark_bubble_reocr
        ) and (
            has_rect_panel_frame or has_dark_oval_bubble
        ):
            direct_crop_result = _run_direct_paddle_candidate_crop_reocr(
                crop,
                idioma_origem=str(page_dict.get("idioma_origem") or page_dict.get("source_language") or "en"),
            )
            direct_scanlation_texts = _candidate_crop_reocr_result_scanlation_credit_texts(direct_crop_result)
            if direct_scanlation_texts:
                rejected_scanlation_credit_texts.extend(direct_scanlation_texts)
                direct_crop_result = {"texts": [], "_vision_blocks": [], "blocks": []}
            if has_dark_oval_bubble and not has_rect_panel_frame:
                for direct_text in list(direct_crop_result.get("texts") or []):
                    if isinstance(direct_text, dict):
                        direct_text["bubble_mask_source"] = "image_dark_bubble_mask"
                        direct_text["block_profile"] = direct_text.get("block_profile") or "dark_bubble"
                        direct_text["layout_profile"] = direct_text.get("layout_profile") or "dark_bubble"
            direct_texts, direct_blocks = _map_crop_ocr_page_to_band(
                direct_crop_result,
                band_page=page_dict,
                band_id=band_id,
                balloon_local_bbox=[x1, y1, x2, y2],
                crop_left=crop_left,
                crop_top=crop_top,
                candidate_index=candidate_index,
            )
            if direct_texts and has_dark_oval_bubble and (allow_dark_bubble_probe or allow_sparse_dark_bubble_reocr):
                direct_texts, direct_blocks = _filter_dark_bubble_reocr_to_balloon(
                    direct_texts,
                    direct_blocks,
                    bubble_bbox=list(dark_oval_bbox or [x1, y1, x2, y2]),
                )
            if direct_texts and has_dark_oval_bubble and (allow_dark_bubble_probe or allow_sparse_dark_bubble_reocr):
                _annotate_dark_bubble_recovery(
                    direct_texts,
                    direct_blocks,
                    bubble_bbox=list(dark_oval_bbox or [x1, y1, x2, y2]),
                )
                texts, blocks = direct_texts, direct_blocks
            elif direct_texts:
                _annotate_dark_panel_recovery(
                    direct_texts,
                    direct_blocks,
                    image=image,
                    panel_bbox=list(rect_panel_bbox or [x1, y1, x2, y2]),
                )
                texts, blocks = direct_texts, direct_blocks
        if not texts and (
            evidence_is_strong
            or high_conf_dark_light_text
        ):
            try:
                dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
                bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
                light_area = int(evidence.get("inner_light_area") or 0)
            except Exception:
                dark_ratio = 0.0
                bright_ratio = 1.0
                light_area = 0
            if dark_ratio >= 0.62 and bright_ratio <= 0.22 and light_area >= 180:
                direct_crop_result = _run_direct_paddle_candidate_crop_reocr(
                    crop,
                    idioma_origem=str(page_dict.get("idioma_origem") or page_dict.get("source_language") or "en"),
                )
                direct_scanlation_texts = _candidate_crop_reocr_result_scanlation_credit_texts(direct_crop_result)
                if direct_scanlation_texts:
                    rejected_scanlation_credit_texts.extend(direct_scanlation_texts)
                    direct_crop_result = {"texts": [], "_vision_blocks": [], "blocks": []}
                direct_texts, direct_blocks = _map_crop_ocr_page_to_band(
                    direct_crop_result,
                    band_page=page_dict,
                    band_id=band_id,
                    balloon_local_bbox=[x1, y1, x2, y2],
                    crop_left=crop_left,
                    crop_top=crop_top,
                    candidate_index=candidate_index,
                )
                if dark_oval_bbox is not None:
                    recovery_bubble_bbox = list(dark_oval_bbox)
                elif high_conf_dark_light_text:
                    recovery_bubble_bbox = [
                        max(0, int(x1) - 24),
                        max(0, int(y1) - 24),
                        min(width, int(x2) + 24),
                        min(height, int(y2) + 24),
                    ]
                else:
                    recovery_bubble_bbox = [x1, y1, x2, y2]
                direct_texts, direct_blocks = _filter_dark_bubble_reocr_to_balloon(
                    direct_texts,
                    direct_blocks,
                    bubble_bbox=recovery_bubble_bbox,
                )
                if direct_texts:
                    _annotate_dark_bubble_recovery(
                        direct_texts,
                        direct_blocks,
                        bubble_bbox=recovery_bubble_bbox,
                    )
                    for item in direct_texts:
                        flags = list(item.get("qa_flags") or [])
                        if "dark_bubble_text_candidate_reocr" not in flags:
                            flags.append("dark_bubble_text_candidate_reocr")
                        if high_conf_dark_light_text and "dark_bubble_high_conf_light_text_reocr" not in flags:
                            flags.append("dark_bubble_high_conf_light_text_reocr")
                        item["qa_flags"] = flags
                    texts, blocks = direct_texts, direct_blocks
        recovered_texts.extend(texts)
        recovered_blocks.extend(blocks)

    page = dict(page_dict)
    page["texts"] = recovered_texts
    page["_vision_blocks"] = recovered_blocks
    page["_ocr_stats"] = {
        "candidate_crop_reocr_candidate_count": int(candidate_count),
        "candidate_crop_reocr_attempts": int(attempts),
        "candidate_crop_reocr_recovered": int(len(recovered_texts)),
    }
    if rejected_scanlation_credit_texts:
        page["_ocr_stats"]["scanlation_discord_promo_detected"] = bool(
            any(
                _scanlation_discord_promo_text_signal(raw)[0]
                and (
                    _scanlation_discord_promo_text_signal(raw)[1]
                    or _scanlation_discord_promo_text_signal(raw)[2]
                )
                for raw in rejected_scanlation_credit_texts
            )
        )
        page["_ocr_stats"]["scanlation_credit_rejected_texts"] = rejected_scanlation_credit_texts[:8]
    return BandStageOutput(
        "ocr_candidate_recovery",
        page,
        {
            "ocr_candidate_crop_candidates": int(candidate_count),
            "ocr_candidate_crop_attempts": int(attempts),
            "ocr_candidate_crop_recovered": int(len(recovered_texts)),
        },
    )


def _recovered_text_overlaps_existing(text: dict, existing_texts: list[dict]) -> bool:
    bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    area = _bbox_area(bbox)
    if bbox is None or area <= 0:
        return False
    recovered_is_dark_lobe = bool(
        _is_dark_bubble_reocr_text(text)
        and "partial_dark_bubble_lobe_reocr" in {str(flag).strip() for flag in text.get("qa_flags") or []}
    )
    bx1, _by1, bx2, _by2 = bbox
    recovered_cx = (bx1 + bx2) / 2.0
    for existing in existing_texts:
        if not isinstance(existing, dict):
            continue
        existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
        existing_area = _bbox_area(existing_bbox)
        if existing_bbox is None or existing_area <= 0:
            continue
        overlap = _bbox_intersection_area(bbox, existing_bbox)
        overlap_ratio = overlap / float(max(1, min(area, existing_area)))
        recovered_text = str(text.get("text") or text.get("original") or text.get("raw_ocr") or "").strip()
        existing_text = str(existing.get("text") or existing.get("original") or existing.get("raw_ocr") or "").strip()
        recovered_compact = re.sub(r"[^A-Za-z0-9]+", "", recovered_text)
        existing_compact = re.sub(r"[^A-Za-z0-9]+", "", existing_text)
        recovered_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9']+", recovered_text)
            if len(token) > 1 or token.lower() in {"a", "i"}
        ]
        existing_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9']+", existing_text)
            if len(token) > 1 or token.lower() in {"a", "i"}
        ]
        if recovered_is_dark_lobe and 3 <= len(recovered_tokens) <= 8 and len(existing_tokens) >= len(recovered_tokens) + 2:
            existing_token_set = set(existing_tokens)
            shared = sum(1 for token in recovered_tokens if token in existing_token_set)
            recovered_balloon = _coerce_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
            existing_balloon = _coerce_bbox(existing.get("balloon_bbox") or existing.get("bubble_mask_bbox"))
            balloon_overlap_ratio = 0.0
            if recovered_balloon is not None and existing_balloon is not None:
                recovered_balloon_area = max(1, _bbox_area(recovered_balloon))
                existing_balloon_area = max(1, _bbox_area(existing_balloon))
                balloon_overlap_ratio = _bbox_intersection_area(recovered_balloon, existing_balloon) / float(
                    max(1, min(recovered_balloon_area, existing_balloon_area))
                )
            if shared / float(max(1, len(recovered_tokens))) >= 0.75 and (
                overlap_ratio >= 0.20 or balloon_overlap_ratio >= 0.35
            ):
                flags = list(text.get("qa_flags") or [])
                if "dark_lobe_semantic_duplicate_fragment_suppressed" not in flags:
                    flags.append("dark_lobe_semantic_duplicate_fragment_suppressed")
                text["qa_flags"] = flags
                return True
        if (
            recovered_is_dark_lobe
            and len(recovered_compact) <= 4
            and len(existing_compact) >= max(8, len(recovered_compact) * 2)
            and overlap_ratio >= 0.30
        ):
            flags = list(text.get("qa_flags") or [])
            if "short_dark_lobe_fragment_covered_by_ocr" not in flags:
                flags.append("short_dark_lobe_fragment_covered_by_ocr")
            text["qa_flags"] = flags
            return True
        if recovered_is_dark_lobe and overlap_ratio < 0.75:
            ex1, _ey1, ex2, _ey2 = existing_bbox
            existing_w = max(1, ex2 - ex1)
            existing_cx = (ex1 + ex2) / 2.0
            if abs(recovered_cx - existing_cx) >= max(32.0, existing_w * 0.20):
                continue
        if overlap_ratio >= 0.45:
            return True
    return False


def _strip_false_trailing_dark_lobe_token(text: dict, recovered_lobe: dict) -> bool:
    if not (_is_dark_bubble_reocr_text(text) and _is_dark_bubble_reocr_text(recovered_lobe)):
        return False
    recovered_value = str(
        recovered_lobe.get("text")
        or recovered_lobe.get("original")
        or recovered_lobe.get("raw_ocr")
        or ""
    )
    if not re.search(r"\b(?:i\s*am|am)\s+called\b|\bsystem\b", recovered_value, re.IGNORECASE):
        return False
    bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    recovered_bbox = _coerce_bbox(
        recovered_lobe.get("text_pixel_bbox") or recovered_lobe.get("bbox") or recovered_lobe.get("source_bbox")
    )
    if bbox is None or recovered_bbox is None:
        return False
    x1, _y1, x2, _y2 = bbox
    rx1, _ry1, _rx2, _ry2 = recovered_bbox
    if rx1 < x1 + int((x2 - x1) * 0.55):
        return False
    changed = False
    for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"):
        value = text.get(key)
        if not isinstance(value, str):
            continue
        cleaned = re.sub(r"(?:\s|[.。])*[IT]an[.!?。]*\s*$", "", value, flags=re.IGNORECASE).strip()
        if cleaned and cleaned != value.strip():
            text[key] = cleaned
            changed = True
    if not changed:
        return False
    polygons = []
    for polygon in list(text.get("line_polygons") or []):
        xs_poly: list[int] = []
        ys_poly: list[int] = []
        for point in polygon:
            try:
                xs_poly.append(int(point[0]))
                ys_poly.append(int(point[1]))
            except Exception:
                continue
        if not xs_poly or not ys_poly:
            polygons.append(polygon)
            continue
        poly_bbox = [min(xs_poly), min(ys_poly), max(xs_poly), max(ys_poly)]
        px1, _py1, px2, _py2 = poly_bbox
        pcx = (px1 + px2) / 2.0
        if pcx >= rx1 - 24:
            continue
        polygons.append(polygon)
    if polygons and len(polygons) != len(list(text.get("line_polygons") or [])):
        text["line_polygons"] = polygons
        xs: list[int] = []
        ys: list[int] = []
        for polygon in polygons:
            for point in polygon:
                try:
                    xs.append(int(point[0]))
                    ys.append(int(point[1]))
                except Exception:
                    continue
        if xs and ys:
            new_bbox = [min(xs), min(ys), max(xs), max(ys)]
            for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
                if key in text:
                    text[key] = list(new_bbox)
    flags = list(text.get("qa_flags") or [])
    if "false_connected_lobe_tail_removed" not in flags:
        flags.append("false_connected_lobe_tail_removed")
    text["qa_flags"] = flags
    return True


def _strip_false_trailing_dark_bubble_fragment(text: dict) -> bool:
    if not _is_dark_bubble_reocr_text(text):
        return False
    raw_value = str(text.get("text") or text.get("original") or text.get("raw_ocr") or "").strip()
    if not raw_value:
        return False
    clipped_match = re.search(r"^(?P<kept>.+?[.!?])\s+(?P<tail>[A-Za-z]{1,3})\s*$", raw_value)
    if clipped_match:
        kept = clipped_match.group("kept").strip()
        tail = clipped_match.group("tail").strip()
        if len(re.sub(r"[^A-Za-z0-9]+", "", kept)) >= 12 and len(tail) <= 3:
            polygons = list(text.get("line_polygons") or [])
            tail_bbox: list[int] | None = None
            if polygons:
                last = polygons[-1]
                xs: list[int] = []
                ys: list[int] = []
                for point in last:
                    try:
                        xs.append(int(point[0]))
                        ys.append(int(point[1]))
                    except Exception:
                        continue
                if xs and ys:
                    tail_bbox = [min(xs), min(ys), max(xs), max(ys)]
            changed = False
            for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"):
                value = text.get(key)
                if not isinstance(value, str):
                    continue
                cleaned = re.sub(r"^(.+?[.!?])\s+[A-Za-z]{1,3}\s*$", r"\1", value.strip()).strip()
                if cleaned and cleaned != value.strip():
                    text[key] = cleaned
                    changed = True
            if changed:
                if tail_bbox is not None:
                    tx1, ty1, tx2, ty2 = tail_bbox
                    cleanup_bbox = [
                        max(0, int(tx1) - 28),
                        max(0, int(ty1) - 8),
                        int(tx2) + 180,
                        int(ty2) + 40,
                    ]
                    cleanup_payload = {
                        "bbox": cleanup_bbox,
                        "removed_tail": tail,
                        "source": "trailing_short_fragment",
                    }
                    text["clipped_overlap_fragment_cleanup_bbox"] = dict(cleanup_payload)
                    metrics = text.setdefault("qa_metrics", {})
                    if isinstance(metrics, dict):
                        metrics["clipped_overlap_fragment_cleanup_bbox"] = dict(cleanup_payload)
                if len(polygons) >= 2:
                    text["line_polygons"] = polygons[:-1]
                    xs_all: list[int] = []
                    ys_all: list[int] = []
                    for polygon in polygons[:-1]:
                        for point in polygon:
                            try:
                                xs_all.append(int(point[0]))
                                ys_all.append(int(point[1]))
                            except Exception:
                                continue
                    if xs_all and ys_all:
                        new_bbox = [min(xs_all), min(ys_all), max(xs_all), max(ys_all)]
                        for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
                            if key in text:
                                text[key] = list(new_bbox)
                flags = list(text.get("qa_flags") or [])
                if "false_dark_bubble_trailing_clipped_fragment_removed" not in flags:
                    flags.append("false_dark_bubble_trailing_clipped_fragment_removed")
                text["qa_flags"] = flags
                return True
    match = re.search(
        r"^(?P<kept>.+?[.!?])\s+(?P<tail>(?:[Ww][Il1I]|[Il1]{1,2})(?:\s+(?:for|of|to|in|is|it))?)\s*$",
        raw_value,
    )
    if not match:
        return False
    kept = match.group("kept").strip()
    tail = match.group("tail").strip()
    if len(re.sub(r"[^A-Za-z0-9]+", "", kept)) < 12 or len(tail.split()) > 2:
        return False
    changed = False
    old_len = max(1, len(raw_value))
    kept_ratio = max(0.55, min(0.98, len(kept) / float(old_len)))
    for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"):
        value = text.get(key)
        if isinstance(value, str):
            cleaned = re.sub(
                r"^(.+?[.!?])\s+(?:[Ww][Il1I]|[Il1]{1,2})(?:\s+(?:for|of|to|in|is|it))?\s*$",
                r"\1",
                value.strip(),
            ).strip()
            if cleaned and cleaned != value.strip():
                text[key] = cleaned
                changed = True
    if not changed:
        return False

    for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
        bbox = _coerce_bbox(text.get(key))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        new_x2 = max(x1 + 1, min(x2, int(round(x1 + width * kept_ratio))))
        if x2 - new_x2 >= 4:
            text[key] = [x1, y1, new_x2, y2]

    polygons = []
    clipped_any = False
    clip_x2 = None
    clipped_bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    if clipped_bbox is not None:
        clip_x2 = int(clipped_bbox[2])
    for polygon in list(text.get("line_polygons") or []):
        if clip_x2 is None:
            polygons.append(polygon)
            continue
        clipped_polygon = []
        for point in polygon:
            try:
                px = int(point[0])
                py = int(point[1])
            except Exception:
                clipped_polygon.append(point)
                continue
            if px > clip_x2:
                clipped_any = True
                px = clip_x2
            clipped_polygon.append([px, py])
        polygons.append(clipped_polygon)
    if clipped_any and polygons:
        text["line_polygons"] = polygons

    flags = list(text.get("qa_flags") or [])
    if "false_dark_bubble_trailing_ocr_fragment_removed" not in flags:
        flags.append("false_dark_bubble_trailing_ocr_fragment_removed")
    text["qa_flags"] = flags
    return True


def _is_dark_bubble_reocr_text(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    profiles = {
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("layout_profile") or "").strip().lower(),
    }
    flags = {str(flag).strip() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    return bool(
        source == "image_dark_bubble_mask"
        or "dark_bubble" in profiles
        or "dark_bubble_oval_reocr" in flags
    )


def _recovered_dark_bubble_replaces_existing(recovered: dict, existing: dict) -> bool:
    if not (_is_dark_bubble_reocr_text(recovered) and isinstance(existing, dict)):
        return False
    if _recovered_dark_bubble_contains_existing_with_extra_lobe_text(recovered, existing):
        return False
    if _recovered_dark_bubble_is_contaminated_by_existing_text(recovered, existing):
        return False
    existing_source = str(existing.get("bubble_mask_source") or existing.get("bubbleMaskSource") or "").strip().lower()
    if _is_dark_bubble_reocr_text(existing) and existing_source == "image_dark_bubble_mask":
        return False
    recovered_bbox = _coerce_bbox(recovered.get("text_pixel_bbox") or recovered.get("bbox") or recovered.get("source_bbox"))
    existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
    if recovered_bbox is None or existing_bbox is None:
        return False
    recovered_area = _bbox_area(recovered_bbox)
    existing_area = _bbox_area(existing_bbox)
    if recovered_area <= 0 or existing_area <= 0:
        return False
    overlap = _bbox_intersection_area(recovered_bbox, existing_bbox)
    if overlap / float(max(1, min(recovered_area, existing_area))) < 0.35:
        return False
    recovered_text = str(recovered.get("text") or recovered.get("original") or "").strip()
    existing_text = str(existing.get("text") or existing.get("original") or "").strip()
    if len(re.sub(r"[^A-Za-z0-9]+", "", recovered_text)) < max(8, len(re.sub(r"[^A-Za-z0-9]+", "", existing_text))):
        return False
    return recovered_area >= int(existing_area * 1.20) or len(recovered_text) >= len(existing_text) + 8


def _compact_ocr_text_for_reocr_compare(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*(?:\d+\s*/\s*|[/\\|]+\s*|\d+\s+)", "", text)
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()


def _recovered_dark_bubble_contains_existing_with_extra_lobe_text(recovered: dict, existing: dict) -> bool:
    if not (_is_dark_bubble_reocr_text(recovered) and isinstance(existing, dict)):
        return False
    recovered_flags = {str(flag).strip() for flag in recovered.get("qa_flags") or [] if str(flag).strip()}
    if "candidate_crop_direct_paddle_reocr" not in recovered_flags:
        return False

    recovered_raw = str(recovered.get("text") or recovered.get("original") or recovered.get("raw_ocr") or "").strip()
    existing_raw = str(existing.get("text") or existing.get("original") or existing.get("raw_ocr") or "").strip()
    recovered_compact = _compact_ocr_text_for_reocr_compare(recovered_raw)
    existing_compact = _compact_ocr_text_for_reocr_compare(existing_raw)
    if len(existing_compact) < 12 or len(recovered_compact) <= len(existing_compact) + 6:
        return False

    existing_index = recovered_compact.find(existing_compact)
    if existing_index < 0:
        return False
    prefix_len = existing_index
    suffix_len = len(recovered_compact) - existing_index - len(existing_compact)
    # Same-lobe reOCR often contains an existing partial OCR plus the missing
    # suffix. Only reject the composite pattern where text from a previous lobe
    # appears before an otherwise complete independent OCR.
    if prefix_len < 6:
        return False
    if suffix_len > max(4, int(round(len(existing_compact) * 0.20))):
        return False

    recovered_bbox = _coerce_bbox(recovered.get("text_pixel_bbox") or recovered.get("bbox") or recovered.get("source_bbox"))
    existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
    if recovered_bbox is None or existing_bbox is None:
        return False

    recovered_area = max(1, _bbox_area(recovered_bbox))
    existing_area = max(1, _bbox_area(existing_bbox))
    existing_covered = _bbox_intersection_area(recovered_bbox, existing_bbox) / float(existing_area)
    if existing_covered < 0.72:
        return False

    _rx1, ry1, _rx2, ry2 = recovered_bbox
    _ex1, ey1, _ex2, ey2 = existing_bbox
    recovered_h = max(1, ry2 - ry1)
    recovered_cy = (ry1 + ry2) / 2.0
    existing_cy = (ey1 + ey2) / 2.0
    vertically_separate = abs(existing_cy - recovered_cy) >= max(24.0, recovered_h * 0.18)
    substantially_broader = recovered_area >= int(existing_area * 2.0)
    if not (vertically_separate or substantially_broader):
        return False

    flags = list(recovered.get("qa_flags") or [])
    if "candidate_crop_composite_contains_independent_ocr" not in flags:
        flags.append("candidate_crop_composite_contains_independent_ocr")
    recovered["qa_flags"] = flags
    return True


def _recovered_dark_bubble_is_contaminated_by_existing_text(recovered: dict, existing: dict) -> bool:
    """Reject dark re-OCR that swallowed SFX/art while a normal text read is already present."""
    if not (_is_dark_bubble_reocr_text(recovered) and isinstance(existing, dict)):
        return False
    recovered_flags = {str(flag).strip() for flag in recovered.get("qa_flags") or [] if str(flag).strip()}
    if "candidate_crop_direct_paddle_reocr" not in recovered_flags:
        return False

    existing_profile = {
        str(existing.get("block_profile") or "").strip().lower(),
        str(existing.get("layout_profile") or "").strip().lower(),
    }
    existing_source = str(existing.get("bubble_mask_source") or existing.get("bubbleMaskSource") or "").strip().lower()
    existing_bg = str(existing.get("background_rgb") or "").strip()
    existing_is_light_balloon = (
        "white_balloon" in existing_profile
        or existing_source in {"image_white_bubble_mask", "derived_white_bubble_mask"}
        or existing_bg in {"[253, 253, 253]", "[254, 254, 254]", "[255, 255, 255]"}
    )
    if not existing_is_light_balloon:
        return False

    recovered_bbox = _coerce_bbox(recovered.get("text_pixel_bbox") or recovered.get("bbox") or recovered.get("source_bbox"))
    existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
    if recovered_bbox is None or existing_bbox is None:
        return False
    recovered_area = _bbox_area(recovered_bbox)
    existing_area = _bbox_area(existing_bbox)
    if recovered_area <= 0 or existing_area <= 0:
        return False
    overlap = _bbox_intersection_area(recovered_bbox, existing_bbox)
    if overlap / float(max(1, existing_area)) < 0.55:
        return False

    recovered_raw = str(recovered.get("text") or recovered.get("original") or recovered.get("raw_ocr") or "").strip()
    existing_raw = str(existing.get("text") or existing.get("original") or existing.get("raw_ocr") or "").strip()
    recovered_compact = _compact_ocr_text_for_reocr_compare(recovered_raw)
    existing_compact = _compact_ocr_text_for_reocr_compare(existing_raw)
    if len(existing_compact) < 5 or len(recovered_compact) < 4:
        return False

    leading_sfx_like = bool(re.match(r"^\s*(?:\d+\s*/|[/\\|]+\s*|[Il1]{1,2}\s*/)", recovered_raw))
    recovered_is_prefix = existing_compact.startswith(recovered_compact) or recovered_compact in existing_compact
    similar_size = len(recovered_compact) <= len(existing_compact) + 2
    overbroad_box = recovered_area >= int(existing_area * 4.0)
    if not (leading_sfx_like or overbroad_box):
        return False
    if not (recovered_is_prefix and similar_size):
        return False

    flags = list(recovered.get("qa_flags") or [])
    if "candidate_crop_reocr_rejected_existing_white_ocr_better" not in flags:
        flags.append("candidate_crop_reocr_rejected_existing_white_ocr_better")
    recovered["qa_flags"] = flags
    return True


def _merge_candidate_crop_recovery_into_ocr_page(ocr_page: dict, recovered_page: dict) -> int:
    if not isinstance(ocr_page, dict) or not isinstance(recovered_page, dict):
        return 0
    existing_texts = [text for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)]
    recovered_texts = [text for text in list(recovered_page.get("texts") or []) if isinstance(text, dict)]
    added_texts: list[dict] = []
    for text in recovered_texts:
        _strip_false_trailing_dark_bubble_fragment(text)
        if _candidate_crop_recovery_is_near_scanlation_credit(text, existing_texts):
            continue
        if any(_recovered_dark_bubble_is_contaminated_by_existing_text(text, existing) for existing in existing_texts):
            continue
        replaced_existing = False
        if _is_dark_bubble_reocr_text(text):
            kept_existing: list[dict] = []
            for existing in existing_texts:
                if _recovered_dark_bubble_replaces_existing(text, existing):
                    replaced_existing = True
                    continue
                _strip_false_trailing_dark_lobe_token(existing, text)
                kept_existing.append(existing)
            existing_texts = kept_existing
            added_texts = [
                added for added in added_texts if not _recovered_dark_bubble_replaces_existing(text, added)
            ]
        if not replaced_existing and _recovered_text_overlaps_existing(text, existing_texts + added_texts):
            continue
        added_texts.append(copy.deepcopy(text))
    if not added_texts:
        return 0

    ocr_page["texts"] = existing_texts + added_texts
    existing_blocks = [block for block in list(ocr_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    recovered_blocks = [block for block in list(recovered_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    block_by_candidate: dict[int, dict] = {}
    for block in recovered_blocks:
        try:
            candidate_index = int(block.get("reocr_candidate_index"))
        except Exception:
            candidate_index = -1
        if candidate_index >= 0 and candidate_index not in block_by_candidate:
            block_by_candidate[candidate_index] = copy.deepcopy(block)
    added_blocks: list[dict] = []
    for text in added_texts:
        try:
            candidate_index = int(text.get("reocr_candidate_index"))
        except Exception:
            candidate_index = -1
        block = block_by_candidate.get(candidate_index)
        if block is not None:
            added_blocks.append(block)
    ocr_page["_vision_blocks"] = existing_blocks + added_blocks
    existing_regions = [region for region in list(ocr_page.get("_bubble_regions") or []) if isinstance(region, dict)]
    added_regions: list[dict] = []
    for text in added_texts:
        bubble_bbox = _coerce_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"))
        if bubble_bbox is None:
            continue
        region = {
            "bbox": list(bubble_bbox),
            "bubble_mask_bbox": list(bubble_bbox),
            "confidence": float(text.get("confidence_raw") or text.get("confidence") or 0.0),
            "bubble_id": str(text.get("bubble_id") or text.get("id") or text.get("text_id") or ""),
            "source": "candidate_crop_reocr_dark_bubble",
        }
        inner = _coerce_bbox(text.get("bubble_inner_bbox"))
        if inner is not None:
            region["bubble_inner_bbox"] = list(inner)
        added_regions.append(region)
    if added_regions:
        ocr_page["_bubble_regions"] = added_regions + existing_regions
    stats = ocr_page.setdefault("_ocr_stats", {})
    if isinstance(stats, dict):
        stats["candidate_crop_reocr_merged_recovered"] = int(stats.get("candidate_crop_reocr_merged_recovered") or 0) + len(added_texts)
    return len(added_texts)


def _text_is_scanlation_credit_suppressed(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if "scanlation_credit_suppressed" in flags:
        return True
    route_reason = str(text.get("route_reason") or text.get("skip_reason") or text.get("suppression_reason") or "").strip().lower()
    if route_reason == "scanlation_credit_suppressed":
        return True
    raw = text.get("text") or text.get("original") or text.get("raw_ocr") or ""
    return _candidate_crop_reocr_text_is_scanlation_credit(raw)


def _candidate_crop_recovery_is_near_scanlation_credit(text: dict, existing_texts: list[dict]) -> bool:
    bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    if bbox is None:
        return False
    for existing in existing_texts:
        if not _text_is_scanlation_credit_suppressed(existing):
            continue
        existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
        if existing_bbox is None:
            continue
        ex1, ey1, ex2, ey2 = [int(v) for v in existing_bbox]
        pad_x = max(60, int(round((ex2 - ex1) * 0.35)))
        pad_y = max(90, int(round((ey2 - ey1) * 0.75)))
        padded = [ex1 - pad_x, ey1 - pad_y, ex2 + pad_x, ey2 + pad_y]
        if _bbox_intersection_area(bbox, padded) > 0:
            return True
    return False


def _negative_evidence_text_bbox(text: dict, fallback_block: dict | None = None) -> list[int] | None:
    for key in ("text_pixel_bbox", "bbox", "source_bbox"):
        bbox = _coerce_bbox(text.get(key))
        if bbox is not None:
            return bbox
    if isinstance(fallback_block, dict):
        return _coerce_bbox(fallback_block.get("bbox"))
    return None


def _negative_evidence_text_confidence(text: dict, fallback_block: dict | None = None) -> float:
    for key in ("confidence_raw", "confidence", "score"):
        try:
            value = float(text.get(key))
            if value > 0:
                return value
        except Exception:
            pass
    if isinstance(fallback_block, dict):
        try:
            return float(fallback_block.get("confidence") or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _negative_candidate_is_partial_edge_noise(text: dict, bbox: list[int], confidence: float) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    if width <= 0 or height <= 0:
        return True
    if height <= 18 and confidence < 0.75:
        return True
    if (x1 < 0 or y1 < 0) and height <= 22 and confidence < 0.80:
        return True
    polygons = text.get("line_polygons")
    if height <= 20 and not polygons and confidence < 0.80:
        return True
    return False


def _negative_candidate_is_suppressed(text: dict) -> bool:
    content_class = str(text.get("content_class") or text.get("tipo") or "").strip().lower()
    if content_class in {"sfx", "onomatopeia", "sound_effect"}:
        return True
    route_reason = str(text.get("route_reason") or text.get("suppression_reason") or "").strip().lower()
    if route_reason in SUPPRESSED_INPAINT_ROUTE_REASONS:
        return True
    route_action = str(text.get("route_action") or "").strip().lower()
    return route_action in {"suppress", "skip", "ignore"}


def _negative_dark_context_metrics(image_rgb: np.ndarray, bbox: list[int]) -> dict | None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.size == 0:
        return None
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = max(18, int(round(bw * 0.45)))
    py = max(18, int(round(bh * 0.75)))
    sx1 = max(0, x1 - px)
    sy1 = max(0, y1 - py)
    sx2 = min(width, x2 + px)
    sy2 = min(height, y2 + py)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = image_rgb[sy1:sy2, sx1:sx2].astype(np.uint8)
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    dark_ratio = float(np.count_nonzero(gray <= 70)) / float(max(1, gray.size))
    bright_ratio = float(np.count_nonzero(gray >= 185)) / float(max(1, gray.size))
    median_luma = float(np.median(gray))
    background_rgb = [int(v) for v in np.median(crop.reshape(-1, 3), axis=0)] if crop.ndim == 3 else [int(median_luma)] * 3
    return {
        "search_bbox": [sx1, sy1, sx2, sy2],
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "median_luma": median_luma,
        "background_rgb": background_rgb,
    }


def _float_metric(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _negative_candidate_matches_existing(candidate: dict, existing: dict) -> bool:
    candidate_boxes = [
        bbox
        for bbox in (
            _coerce_bbox(candidate.get("text_pixel_bbox")),
            _coerce_bbox(candidate.get("bbox")),
            _coerce_bbox(candidate.get("source_bbox")),
        )
        if bbox is not None
    ]
    existing_boxes = [
        bbox
        for bbox in (
            _coerce_bbox(existing.get("text_pixel_bbox")),
            _coerce_bbox(existing.get("bbox")),
            _coerce_bbox(existing.get("source_bbox")),
        )
        if bbox is not None
    ]
    if not candidate_boxes or not existing_boxes:
        return False
    candidate_text = candidate.get("text") or candidate.get("original") or candidate.get("raw_ocr") or ""
    existing_text = existing.get("text") or existing.get("original") or existing.get("raw_ocr") or ""
    for candidate_bbox in candidate_boxes:
        for existing_bbox in existing_boxes:
            if _bbox_iou(candidate_bbox, existing_bbox) >= 0.65:
                return True
            if _bbox_overlap_min_ratio(candidate_bbox, existing_bbox) >= 0.55:
                return True
            if _bbox_center_distance(candidate_bbox, existing_bbox) <= 16:
                return _text_similarity(candidate_text, existing_text) >= 0.85
    return False


def _attach_negative_evidence_to_existing(existing: dict, candidate: dict) -> None:
    metrics = existing.setdefault("qa_metrics", {})
    if not isinstance(metrics, dict):
        return
    candidate_bbox = _coerce_bbox(candidate.get("bbox") or candidate.get("text_pixel_bbox") or candidate.get("source_bbox"))
    existing_bbox = _coerce_bbox(existing.get("text_pixel_bbox") or existing.get("bbox") or existing.get("source_bbox"))
    candidate_text = str(candidate.get("text") or candidate.get("original") or "").strip()
    existing_text = str(existing.get("text") or existing.get("original") or "").strip()
    attached = metrics.setdefault("negative_evidence", [])
    if not isinstance(attached, list):
        attached = []
        metrics["negative_evidence"] = attached
    note_like_negative = bool(re.match(r"^\s*T\s*/\s*N\s*[:：]", candidate_text, flags=re.IGNORECASE))
    if candidate_bbox is not None and note_like_negative:
        x1, y1, x2, y2 = [int(v) for v in candidate_bbox]
        candidate_bbox = [x1 - 40, y1 - 14, x2 + 48, y2 + 18]
    attached.append(
        {
            "source": "negative_detect_ocr",
            "bbox": list(candidate_bbox or []),
            "text": candidate_text[:160],
            "confidence": candidate.get("confidence"),
        }
    )
    if candidate_bbox is None or existing_bbox is None or not candidate_text or not existing_text:
        return
    candidate_compact = re.sub(r"[^0-9A-Za-z]+", "", candidate_text).lower()
    existing_compact = re.sub(r"[^0-9A-Za-z]+", "", existing_text).lower()
    if not existing_compact or existing_compact not in candidate_compact:
        return
    candidate_area = max(1, (int(candidate_bbox[2]) - int(candidate_bbox[0])) * (int(candidate_bbox[3]) - int(candidate_bbox[1])))
    existing_area = max(1, (int(existing_bbox[2]) - int(existing_bbox[0])) * (int(existing_bbox[3]) - int(existing_bbox[1])))
    if candidate_area < max(existing_area * 1.45, existing_area + 900):
        return
    if _bbox_overlap_min_ratio(candidate_bbox, existing_bbox) < 0.35:
        return
    flags = list(existing.get("qa_flags") or [])
    lower_flags = {str(flag).strip().lower() for flag in flags if str(flag).strip()}
    source = str(existing.get("bubble_mask_source") or existing.get("bubbleMaskSource") or "").strip().lower()
    dark_connected = (
        source == "image_dark_bubble_mask"
        or "dark_bubble_negative_evidence" in lower_flags
        or "dark_bubble_connected_lobes_promoted" in lower_flags
        or "dark_bubble_lobe_mask_bbox_preferred" in lower_flags
    )
    suffix_len = len(candidate_compact) - len(existing_compact)
    if (
        dark_connected
        and not note_like_negative
        and candidate_compact.startswith(existing_compact)
        and suffix_len >= 6
        and len(existing_text.split()) >= 6
    ):
        if "negative_broad_prefix_candidate_rejected" not in flags:
            flags.append("negative_broad_prefix_candidate_rejected")
        existing["qa_flags"] = flags
        return
    for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"):
        if key in existing or key in {"text", "original"}:
            existing[key] = candidate_text
    for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
        existing[key] = list(candidate_bbox)
    existing["line_polygons"] = [
        [
            [int(candidate_bbox[0]), int(candidate_bbox[1])],
            [int(candidate_bbox[2]), int(candidate_bbox[1])],
            [int(candidate_bbox[2]), int(candidate_bbox[3])],
            [int(candidate_bbox[0]), int(candidate_bbox[3])],
        ]
    ]
    existing["ocr_source"] = "negative_detect_ocr_attached_promoted"
    for flag in ("negative_pass_attached_promoted", "dark_bubble_negative_evidence"):
        if flag not in flags:
            flags.append(flag)
    existing["qa_flags"] = flags


def _attach_dark_bubble_contract_from_negative_duplicate(
    existing: dict,
    candidate: dict,
    *,
    text_bbox: list[int],
    confidence: float,
    image_rgb: np.ndarray,
    context: dict,
    index: int,
) -> bool:
    """Upgrade a normal-pass duplicate when the negative pass found the real dark bubble."""
    if not isinstance(existing, dict):
        return False
    source = str(existing.get("bubble_mask_source") or existing.get("bubbleMaskSource") or "").strip().lower()
    stale_sources = {
        "",
        "bbox_fallback",
        "balloon_bbox_fallback",
        "derived_white_crop_rejected",
        "rejected_derived_bubble_mask",
        "image_white_bubble_mask",
        "image_rect_bubble_mask",
        "image_contour_bubble_mask",
    }
    if source not in stale_sources:
        return False
    flags = {str(flag).strip().lower() for flag in existing.get("qa_flags") or [] if str(flag).strip()}
    if source not in {"image_white_bubble_mask", "image_rect_bubble_mask", "image_contour_bubble_mask"} and (
        "dark_bubble_negative_evidence" not in flags
    ):
        return False
    built = _build_negative_dark_candidate(
        candidate,
        text_bbox=text_bbox,
        confidence=confidence or 0.50,
        image_rgb=image_rgb,
        context=context,
        index=index,
    )
    if built is None:
        return False
    promoted, _block, _region = built
    promoted_source = str(promoted.get("bubble_mask_source") or "").strip().lower()
    if promoted_source != "image_dark_bubble_mask":
        return False
    for key in (
        "bubble_mask_source",
        "bubble_mask_bbox",
        "balloon_bbox",
        "bubble_mask_shape",
        "bubble_mask_ellipse",
        "block_profile",
        "layout_profile",
        "background_rgb",
    ):
        if promoted.get(key) is not None:
            existing[key] = copy.deepcopy(promoted[key])
    metrics = existing.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        promoted_metrics = promoted.get("qa_metrics") if isinstance(promoted.get("qa_metrics"), dict) else {}
        if isinstance(promoted_metrics.get("image_dark_bubble_mask"), dict):
            metrics["image_dark_bubble_mask"] = copy.deepcopy(promoted_metrics["image_dark_bubble_mask"])
        else:
            metrics["image_dark_bubble_mask"] = {
                "source": "image_dark_bubble_mask",
                "detection_space": "negative_duplicate_dark_contract",
                "mask_bbox": list(promoted.get("bubble_mask_bbox") or []),
                "anchor_bbox": list(text_bbox),
            }
    updated_flags = list(existing.get("qa_flags") or [])
    for flag in ("dark_bubble_duplicate_contract_promoted", "dark_bubble_ellipse_bbox_mask"):
        if flag not in updated_flags:
            updated_flags.append(flag)
    existing["qa_flags"] = updated_flags
    existing.pop("bubble_mask_error", None)
    existing.pop("bubbleMaskError", None)
    return True


def _build_negative_dark_candidate(
    text: dict,
    *,
    text_bbox: list[int],
    confidence: float,
    image_rgb: np.ndarray,
    context: dict,
    index: int,
) -> tuple[dict, dict, dict | None] | None:
    height, width = image_rgb.shape[:2]
    probe_bbox = context.get("search_bbox") or text_bbox
    bubble_bbox = _detect_dark_oval_bubble_bbox(
        image_rgb,
        BBox(int(probe_bbox[0]), int(probe_bbox[1]), int(probe_bbox[2]), int(probe_bbox[3])),
    )
    if bubble_bbox is not None:
        bubble_source = "image_dark_bubble_mask"
        profile = "dark_bubble"
        shape = "ellipse"
    else:
        sx1, sy1, sx2, sy2 = [int(v) for v in probe_bbox]
        if _float_metric(context.get("dark_ratio"), 0.0) < 0.48 or _float_metric(context.get("bright_ratio"), 1.0) > 0.24:
            return None
        bubble_bbox = [max(0, sx1), max(0, sy1), min(width, sx2), min(height, sy2)]
        bubble_source = "image_dark_panel_mask"
        profile = "dark_panel"
        shape = "rect"
    raw_text = str(text.get("text") or text.get("original") or text.get("raw_ocr") or "").strip()
    promoted = copy.deepcopy(text)
    promoted.update(
        {
            "id": promoted.get("id") or f"negative_dark_{index:03d}",
            "text": raw_text,
            "original": promoted.get("original") or raw_text,
            "bbox": list(text_bbox),
            "source_bbox": list(text_bbox),
            "text_pixel_bbox": list(text_bbox),
            "confidence": confidence,
            "confidence_raw": confidence,
            "ocr_source": "negative_detect_ocr_promoted",
            "bubble_mask_source": bubble_source,
            "bubble_mask_bbox": list(bubble_bbox),
            "balloon_bbox": list(bubble_bbox),
            "bubble_mask_shape": shape,
            "block_profile": profile,
            "layout_profile": profile,
            "background_rgb": context.get("background_rgb"),
        }
    )
    flags = list(promoted.get("qa_flags") or [])
    for flag in ("negative_pass_promoted", "dark_bubble_negative_evidence"):
        if flag not in flags:
            flags.append(flag)
    if profile == "dark_bubble" and "dark_bubble_oval_reocr" not in flags:
        flags.append("dark_bubble_oval_reocr")
    promoted["qa_flags"] = flags
    metrics = promoted.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["negative_evidence"] = {
            "source": "negative_detect_ocr",
            "image_transform": "inverted_luma",
            "search_bbox": list(context.get("search_bbox") or []),
            "dark_ratio": context.get("dark_ratio"),
            "bright_ratio": context.get("bright_ratio"),
            "median_luma": context.get("median_luma"),
        }
    block = {
        "bbox": list(text_bbox),
        "confidence": confidence,
        "detector": "negative_detect_ocr_promoted",
        "ocr_source": "negative_detect_ocr_promoted",
        "bubble_mask_source": bubble_source,
        "bubble_mask_bbox": list(bubble_bbox),
        "balloon_bbox": list(bubble_bbox),
        "block_profile": profile,
        "layout_profile": profile,
    }
    region = None
    if profile == "dark_bubble":
        x1, y1, x2, y2 = [int(v) for v in bubble_bbox]
        region = {
            "bbox": list(bubble_bbox),
            "bubble_mask_bbox": list(bubble_bbox),
            "confidence": confidence,
            "bubble_id": str(promoted.get("id") or f"negative_dark_{index:03d}"),
            "source": "negative_detect_ocr_promoted",
        }
        promoted["bubble_mask_ellipse"] = {
            "center": [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)],
            "axes": [float(max(1, x2 - x1)), float(max(1, y2 - y1))],
            "angle": 0.0,
        }
    return promoted, block, region


def fuse_negative_dark_bubble_candidates(normal_page: dict, negative_evidence: dict | None, image_rgb: np.ndarray) -> int:
    if not isinstance(normal_page, dict) or not isinstance(negative_evidence, dict):
        return 0
    texts = [text for text in list(normal_page.get("texts") or []) if isinstance(text, dict)]
    evidence_texts = [text for text in list(negative_evidence.get("texts") or []) if isinstance(text, dict)]
    evidence_blocks = [block for block in list(negative_evidence.get("blocks") or []) if isinstance(block, dict)]
    if not evidence_texts:
        return 0
    added_texts: list[dict] = []
    added_blocks: list[dict] = []
    added_regions: list[dict] = []
    attached = 0
    evidence_items: list[tuple[int, dict, dict | None, list[int] | None, int]] = []
    for original_index, evidence_text in enumerate(evidence_texts):
        fallback_block = evidence_blocks[original_index] if original_index < len(evidence_blocks) else None
        precomputed_bbox = _negative_evidence_text_bbox(evidence_text, fallback_block)
        evidence_items.append(
            (
                original_index,
                evidence_text,
                fallback_block,
                precomputed_bbox,
                _bbox_area(precomputed_bbox) if precomputed_bbox is not None else 10**12,
            )
        )
    evidence_items.sort(key=lambda item: (item[4], item[0]))
    for index, text, fallback_block, text_bbox, _area in evidence_items:
        if text_bbox is None:
            continue
        confidence = _negative_evidence_text_confidence(text, fallback_block)
        if confidence and confidence < 0.28:
            continue
        if _negative_candidate_is_partial_edge_noise(text, text_bbox, confidence):
            continue
        if _negative_candidate_is_suppressed(text):
            continue
        if not _candidate_crop_reocr_text_is_usable(text):
            continue
        context = _negative_dark_context_metrics(image_rgb, text_bbox)
        if not context:
            continue
        if _float_metric(context.get("dark_ratio"), 0.0) < 0.35 or _float_metric(context.get("bright_ratio"), 1.0) > 0.34:
            continue
        candidate = copy.deepcopy(text)
        candidate["bbox"] = list(text_bbox)
        candidate["text_pixel_bbox"] = list(text_bbox)
        duplicate = None
        for existing in texts + added_texts:
            if isinstance(existing, dict) and _negative_candidate_matches_existing(candidate, existing):
                duplicate = existing
                break
        if duplicate is not None:
            _attach_negative_evidence_to_existing(duplicate, candidate)
            _attach_dark_bubble_contract_from_negative_duplicate(
                duplicate,
                candidate,
                text_bbox=text_bbox,
                confidence=confidence or 0.50,
                image_rgb=image_rgb,
                context=context,
                index=index,
            )
            attached += 1
            continue
        built = _build_negative_dark_candidate(
            text,
            text_bbox=text_bbox,
            confidence=confidence or 0.50,
            image_rgb=image_rgb,
            context=context,
            index=index,
        )
        if built is None:
            continue
        promoted, block, region = built
        added_texts.append(promoted)
        added_blocks.append(block)
        if region is not None:
            added_regions.append(region)
    if added_texts or attached:
        normal_page["texts"] = texts + added_texts
    if added_texts:
        normal_page["_vision_blocks"] = [
            block for block in list(normal_page.get("_vision_blocks") or []) if isinstance(block, dict)
        ] + added_blocks
        if added_regions:
            normal_page["_bubble_regions"] = added_regions + [
                region for region in list(normal_page.get("_bubble_regions") or []) if isinstance(region, dict)
            ]
    stats = normal_page.setdefault("_ocr_stats", {})
    if isinstance(stats, dict):
        if added_texts:
            stats["negative_dark_candidates_promoted"] = int(stats.get("negative_dark_candidates_promoted") or 0) + len(added_texts)
        if attached:
            stats["negative_dark_candidates_attached"] = int(stats.get("negative_dark_candidates_attached") or 0) + attached
    return len(added_texts)


def _recover_partial_dark_bubble_ocr_from_texts(
    ocr_page: dict,
    *,
    band: Band,
    page_dict: dict,
    band_id: str,
    idioma_origem: str = "en",
    runtime=None,
    work_title: str = "",
    work_title_user_provided: bool = False,
    layout_lobes_only: bool = False,
) -> int:
    if not isinstance(ocr_page, dict):
        return 0
    try:
        from strip.detect_balloons import _inner_dark_text_evidence
    except Exception:
        _inner_dark_text_evidence = None
    if _inner_dark_text_evidence is None:
        return 0
    image = band.strip_slice
    if not isinstance(image, np.ndarray) or image.size == 0:
        return 0
    height, width = image.shape[:2]
    recovered_page = {"texts": [], "_vision_blocks": []}
    debug_stats = {
        "dark_texts": 0,
        "lobe_candidates": 0,
        "lobe_evidence_passed": 0,
        "lobe_ocr_attempts": 0,
        "lobe_ocr_recovered": 0,
        "low_confidence_attempts": 0,
        "layout_connected_lobe_candidates": 0,
        "layout_connected_lobe_recovered": 0,
        "adjacent_dark_bubble_candidates": 0,
        "adjacent_dark_bubble_recovered": 0,
    }
    candidate_index = 1000

    def _call_runtime_crop(crop: np.ndarray, crop_page: dict) -> dict:
        if runtime is None:
            return {"texts": [], "_vision_blocks": []}
        if work_title or work_title_user_provided:
            try:
                return runtime.run_ocr_stage(
                    crop,
                    crop_page,
                    work_title=work_title,
                    work_title_user_provided=work_title_user_provided,
                )
            except TypeError as exc:
                message = str(exc)
                if "unexpected keyword argument" not in message and "got an unexpected keyword" not in message:
                    raise
        return runtime.run_ocr_stage(crop, crop_page)

    def _append_runtime_recovered_lobe(
        text: dict,
        crop_bbox: list[int],
        bubble_bbox: list[int],
        *,
        output_page_space: bool = False,
        trim_leading_edge_fragments: bool = False,
    ) -> bool:
        nonlocal candidate_index
        debug_stats["lobe_ocr_attempts"] += 1
        ocr_crop_bbox = _tight_light_text_crop_bbox_for_dark_lobe(image, crop_bbox)
        crop_left, crop_top, crop_right, crop_bottom = [int(v) for v in ocr_crop_bbox]
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return False
        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        crop_page = dict(page_dict)
        crop_page.update(
            {
                "width": int(crop.shape[1]),
                "height": int(crop.shape[0]),
                "_band_id": band_id,
                "_candidate_crop_reocr": True,
                "_candidate_crop_offset": [int(crop_left), int(crop_top)],
                "_vision_blocks": [
                    {
                        "bbox": [0, 0, int(crop.shape[1]), int(crop.shape[0])],
                        "confidence": float(text.get("confidence_raw") or text.get("confidence") or 0.0),
                        "band_id": band_id,
                        "bubble_id": f"{band_id}_partial_dark_lobe_{candidate_index:03d}",
                        "balloon_bbox": [0, 0, int(crop.shape[1]), int(crop.shape[0])],
                        "bubble_mask_bbox": [0, 0, int(crop.shape[1]), int(crop.shape[0])],
                    }
                ],
                "_bubble_regions": [],
            }
        )
        def _map_lobe_page(crop_ocr_page: dict) -> tuple[list[dict], list[dict]]:
            if not isinstance(crop_ocr_page, dict) or not list(crop_ocr_page.get("texts") or []):
                return [], []
            for recovered_text in list(crop_ocr_page.get("texts") or []):
                if not isinstance(recovered_text, dict):
                    continue
                recovered_text["bubble_mask_source"] = "image_dark_bubble_mask"
                recovered_text["block_profile"] = recovered_text.get("block_profile") or "dark_bubble"
                recovered_text["layout_profile"] = recovered_text.get("layout_profile") or "dark_bubble"
                flags = list(recovered_text.get("qa_flags") or [])
                if "dark_bubble_oval_reocr" not in flags:
                    flags.append("dark_bubble_oval_reocr")
                recovered_text["qa_flags"] = flags
            mapped_texts, mapped_blocks = _map_crop_ocr_page_to_band(
                crop_ocr_page,
                band_page=page_dict,
                band_id=band_id,
                balloon_local_bbox=bubble_bbox,
                crop_left=crop_left,
                crop_top=crop_top,
                candidate_index=candidate_index,
            )
            mapped_texts = [
                item for item in mapped_texts if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            return mapped_texts, mapped_blocks

        lang = str(page_dict.get("idioma_origem") or page_dict.get("source_language") or "en")
        direct_texts, direct_blocks = _map_lobe_page(_call_runtime_crop(crop, crop_page))
        if not direct_texts:
            direct_page = _run_direct_paddle_candidate_crop_reocr(crop, idioma_origem=lang)
            direct_texts, direct_blocks = _map_lobe_page(direct_page)
        if not direct_texts and isinstance(crop, np.ndarray) and crop.ndim == 3 and crop.shape[2] >= 3:
            swapped = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2RGB)
            direct_texts, direct_blocks = _map_lobe_page(
                _run_direct_paddle_candidate_crop_reocr(swapped, idioma_origem=lang)
            )
        if not direct_texts:
            negative_crop = _negative_contrast_ocr_crop(crop)
            if negative_crop is not None:
                negative_crop_page = dict(crop_page)
                negative_crop_page["_candidate_crop_negative_contrast_reocr"] = True
                direct_texts, direct_blocks = _map_lobe_page(_call_runtime_crop(negative_crop, negative_crop_page))
                if not direct_texts:
                    direct_texts, direct_blocks = _map_lobe_page(
                        _run_direct_paddle_candidate_crop_reocr(negative_crop, idioma_origem=lang)
                    )
        if not direct_texts:
            return False
        direct_texts, direct_blocks = _filter_dark_bubble_reocr_to_balloon(
            direct_texts,
            direct_blocks,
            bubble_bbox=bubble_bbox,
        )
        if not direct_texts:
            return False
        _annotate_dark_bubble_recovery(direct_texts, direct_blocks, bubble_bbox=bubble_bbox)
        if trim_leading_edge_fragments:
            bx1, _by1, bx2, _by2 = [int(v) for v in bubble_bbox]
            min_keep_cx = bx1 + max(36.0, (bx2 - bx1) * 0.25)
            for item in direct_texts:
                if not isinstance(item, dict):
                    continue
                raw_value = str(item.get("text") or item.get("raw_ocr") or item.get("original") or "")
                called_match = re.search(r"([I1lT]?\s*a?m\s+called\b.*)", raw_value, flags=re.IGNORECASE)
                if called_match:
                    cleaned = called_match.group(1).strip()
                    cleaned = re.sub(r"^[1lT]\s*", "I ", cleaned, flags=re.IGNORECASE)
                    cleaned = re.sub(r"^I\s*a?m\b", "I am", cleaned, flags=re.IGNORECASE)
                    for key in ("text", "raw_ocr", "original", "normalized_ocr", "normalized_text_final"):
                        if key in item:
                            item[key] = cleaned
                polygons = list(item.get("line_polygons") or [])
                if polygons:
                    kept_polygons = []
                    kept_texts = []
                    raw_line_texts = list(item.get("line_texts") or [])
                    for index, polygon in enumerate(polygons):
                        xs: list[int] = []
                        ys: list[int] = []
                        for point in polygon:
                            try:
                                xs.append(int(point[0]))
                                ys.append(int(point[1]))
                            except Exception:
                                continue
                        if not xs or not ys:
                            continue
                        line_value = str(raw_line_texts[index] if index < len(raw_line_texts) else "")
                        if ((min(xs) + max(xs)) / 2.0) >= min_keep_cx or re.search(
                            r"\bcalled\b|\bsystem\b",
                            line_value,
                            flags=re.IGNORECASE,
                        ):
                            kept_polygons.append(polygon)
                            if index < len(raw_line_texts):
                                kept_texts.append(raw_line_texts[index])
                    if kept_polygons and len(kept_polygons) < len(polygons):
                        item["line_polygons"] = kept_polygons
                        if kept_texts:
                            item["line_texts"] = kept_texts
                        xs = []
                        ys = []
                        for polygon in kept_polygons:
                            for point in polygon:
                                try:
                                    xs.append(int(point[0]))
                                    ys.append(int(point[1]))
                                except Exception:
                                    continue
                        if xs and ys:
                            trimmed_bbox = [min(xs), min(ys), max(xs), max(ys)]
                            for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
                                if key in item:
                                    item[key] = list(trimmed_bbox)
        if output_page_space:
            band_top = int(getattr(band, "y_top", 0) or 0)

            def _shift_item_to_page_space(item: dict) -> None:
                for key in (
                    "bbox",
                    "source_bbox",
                    "text_pixel_bbox",
                    "balloon_bbox",
                    "layout_bbox",
                    "bubble_mask_bbox",
                    "bubble_inner_bbox",
                ):
                    shifted = _shift_bbox_xy(item.get(key), 0, band_top)
                    if shifted is not None:
                        item[key] = shifted
                if "line_polygons" in item:
                    item["line_polygons"] = _shift_line_polygons_xy(item.get("line_polygons"), 0, band_top)

            for item in direct_texts:
                if isinstance(item, dict):
                    _shift_item_to_page_space(item)
            for item in direct_blocks:
                if isinstance(item, dict):
                    _shift_item_to_page_space(item)
        for item in direct_texts:
            flags = list(item.get("qa_flags") or [])
            if "partial_dark_bubble_lobe_reocr" not in flags:
                flags.append("partial_dark_bubble_lobe_reocr")
            item["qa_flags"] = flags
        recovered_page["texts"].extend(direct_texts)
        recovered_page["_vision_blocks"].extend(direct_blocks)
        debug_stats["lobe_ocr_recovered"] += len(direct_texts)
        candidate_index += 1
        return True

    def _recover_uncovered_dark_lobes(text: dict, bbox: list[int]) -> None:
        def _localize_bbox(raw_bbox: list[int] | None) -> list[int] | None:
            localized = _coerce_bbox(raw_bbox)
            if localized is None:
                return None
            x1, y1, x2, y2 = [int(v) for v in localized]
            if x2 <= x1 or y2 <= y1:
                return None
            if y1 >= height or y2 > height:
                band_top = int(getattr(band, "y_top", 0) or 0)
                shifted = [x1, y1 - band_top, x2, y2 - band_top]
                sx1, sy1, sx2, sy2 = shifted
                if sx2 > sx1 and sy2 > sy1 and sy2 > 0 and sy1 < height:
                    localized = shifted
            return localized

        bbox = _localize_bbox(bbox) or bbox
        bubble_bbox = (
            _localize_bbox(text.get("bubble_mask_bbox"))
            or _localize_bbox(text.get("balloon_bbox"))
        )
        if bubble_bbox is None:
            tx1, ty1, tx2, ty2 = [int(v) for v in bbox]
            probe_top = max(0, ty1 - max(120, int((ty2 - ty1) * 1.25)))
            probe_bottom = min(height, ty2 + max(120, int((ty2 - ty1) * 1.25)))
            detected = _detect_dark_oval_bubble_bbox(
                image,
                BBox(0, probe_top, width, probe_bottom),
            )
            if detected is not None:
                bubble_bbox = detected
        if bubble_bbox is None:
            return
        bx1, by1, bx2, by2 = [int(v) for v in bubble_bbox]
        tx1, ty1, tx2, ty2 = [int(v) for v in bbox]
        bx1 = max(0, min(width, bx1))
        bx2 = max(0, min(width, bx2))
        by1 = max(0, min(height, by1))
        by2 = max(0, min(height, by2))
        if bx2 <= bx1 or by2 <= by1:
            return
        bubble_w = bx2 - bx1
        bubble_h = by2 - by1
        text_w = max(1, tx2 - tx1)
        if bubble_w < max(260, int(text_w * 1.85)) or bubble_h < 90:
            return
        min_lobe_w = max(130, int(round(bubble_w * 0.22)))
        candidates: list[list[int]] = []
        left_w = max(0, tx1 - bx1)
        right_w = max(0, bx2 - tx2)
        if left_w >= min_lobe_w:
            candidates.append([bx1, by1, min(bx2, tx1), by2])
        if right_w >= min_lobe_w:
            candidates.append([max(bx1, tx2), by1, bx2, by2])
        debug_stats["lobe_candidates"] += len(candidates)
        for candidate in candidates:
            evidence = _inner_dark_text_evidence(
                image,
                BBox(candidate[0], candidate[1], candidate[2], candidate[3]),
            )
            if not (
                _dark_bubble_evidence_supports_lobe_reocr(
                    evidence,
                    min_dark_ratio=0.32,
                    max_bright_ratio=0.28,
                    min_light_area=300,
                )
            ):
                continue
            debug_stats["lobe_evidence_passed"] += 1
            _append_runtime_recovered_lobe(text, candidate, candidate)

    def _recover_layout_connected_lobes(text: dict, bbox: list[int]) -> None:
        def _localize_bbox(raw_bbox: list[int] | None) -> list[int] | None:
            localized = _coerce_bbox(raw_bbox)
            if localized is None:
                return None
            x1, y1, x2, y2 = [int(v) for v in localized]
            if x2 <= x1 or y2 <= y1:
                return None
            if y1 >= height or y2 > height:
                band_top = int(getattr(band, "y_top", 0) or 0)
                shifted = [x1, y1 - band_top, x2, y2 - band_top]
                sx1, sy1, sx2, sy2 = shifted
                if sx2 > sx1 and sy2 > sy1 and sy2 > 0 and sy1 < height:
                    localized = shifted
            lx1, ly1, lx2, ly2 = [int(v) for v in localized]
            return [
                max(0, min(width, lx1)),
                max(0, min(height, ly1)),
                max(0, min(width, lx2)),
                max(0, min(height, ly2)),
            ]

        localized_text_bbox = _localize_bbox(bbox)
        if localized_text_bbox is None:
            return
        tx1, _ty1, tx2, _ty2 = localized_text_bbox
        text_w = max(1, tx2 - tx1)
        text_cx = (tx1 + tx2) / 2.0
        raw_lobes = text.get("connected_position_bboxes")
        if not isinstance(raw_lobes, list) or len(raw_lobes) < 2:
            return
        for raw_lobe in raw_lobes:
            lobe = _localize_bbox(raw_lobe)
            if lobe is None:
                continue
            lx1, ly1, lx2, ly2 = [int(v) for v in lobe]
            lobe_w = lx2 - lx1
            lobe_h = ly2 - ly1
            if lobe_w < 70 or lobe_h < 50:
                continue
            lobe_cx = (lx1 + lx2) / 2.0
            if abs(lobe_cx - text_cx) < max(38.0, text_w * 0.18):
                continue
            debug_stats["layout_connected_lobe_candidates"] += 1
            evidence = _inner_dark_text_evidence(image, BBox(lx1, ly1, lx2, ly2))
            if not _dark_bubble_evidence_supports_lobe_reocr(
                evidence,
                min_dark_ratio=0.30,
                max_bright_ratio=0.30,
                min_light_area=120,
            ):
                continue
            before = len(recovered_page["texts"])
            output_page_space = bool((_coerce_bbox(text.get("bbox")) or [0, 0, 0, 0])[1] >= height)
            if _append_runtime_recovered_lobe(text, lobe, lobe, output_page_space=output_page_space):
                for recovered_text in recovered_page["texts"][before:]:
                    if isinstance(recovered_text, dict):
                        flags = list(recovered_text.get("qa_flags") or [])
                        if "layout_connected_lobe_reocr" not in flags:
                            flags.append("layout_connected_lobe_reocr")
                        recovered_text["qa_flags"] = flags
                debug_stats["layout_connected_lobe_recovered"] += len(recovered_page["texts"]) - before

    def _recover_adjacent_dark_bubbles(text: dict, bbox: list[int]) -> None:
        def _localize_bbox(raw_bbox: list[int] | None) -> list[int] | None:
            localized = _coerce_bbox(raw_bbox)
            if localized is None:
                return None
            x1, y1, x2, y2 = [int(v) for v in localized]
            if y1 >= height or y2 > height:
                band_top = int(getattr(band, "y_top", 0) or 0)
                shifted = [x1, y1 - band_top, x2, y2 - band_top]
                sx1, sy1, sx2, sy2 = shifted
                if sx2 > sx1 and sy2 > sy1 and sy2 > 0 and sy1 < height:
                    x1, y1, x2, y2 = shifted
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))
            if x2 <= x1 or y2 <= y1:
                return None
            return [x1, y1, x2, y2]

        base = (
            _localize_bbox(text.get("balloon_bbox"))
            or _localize_bbox(text.get("bubble_mask_bbox"))
            or _localize_bbox(bbox)
        )
        if base is None:
            return
        bx1, by1, bx2, by2 = [int(v) for v in base]
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        probes: list[list[int]] = []
        if bx2 < width - 80:
            probes.append(
                [
                    max(0, bx2 - max(72, int(round(bw * 0.25)))),
                    max(0, by1 - max(56, int(round(bh * 0.35)))),
                    width,
                    min(height, by2 + max(96, int(round(bh * 0.85)))),
                ]
            )
        if bx1 > 80:
            probes.append(
                [
                    0,
                    max(0, by1 - max(56, int(round(bh * 0.35)))),
                    min(width, bx1 + max(72, int(round(bw * 0.25)))),
                    min(height, by2 + max(96, int(round(bh * 0.85)))),
                ]
            )
        existing_bboxes = [
            _coerce_bbox(item.get("text_pixel_bbox") or item.get("bbox") or item.get("source_bbox"))
            for item in list(ocr_page.get("texts") or []) + list(recovered_page.get("texts") or [])
            if isinstance(item, dict)
        ]
        for probe in probes:
            px1, py1, px2, py2 = [int(v) for v in probe]
            if px2 <= px1 or py2 <= py1:
                continue
            detected = _detect_dark_oval_bubble_bbox(image, BBox(px1, py1, px2, py2))
            candidate = _localize_bbox(detected)
            if candidate is None:
                continue
            if _bbox_intersection_area(candidate, base) / float(max(1, min(_bbox_area(candidate), _bbox_area(base)))) >= 0.35:
                continue
            candidate_area = _bbox_area(candidate)
            if candidate_area <= 0:
                continue
            covered = False
            for existing_bbox in existing_bboxes:
                if existing_bbox is None:
                    continue
                overlap = _bbox_intersection_area(candidate, existing_bbox)
                if overlap / float(max(1, min(candidate_area, _bbox_area(existing_bbox)))) >= 0.40:
                    covered = True
                    break
            if covered:
                continue
            cx1, cy1, cx2, cy2 = [int(v) for v in candidate]
            evidence = _inner_dark_text_evidence(image, BBox(cx1, cy1, cx2, cy2))
            if not _dark_bubble_evidence_supports_lobe_reocr(
                evidence,
                min_dark_ratio=0.32,
                max_bright_ratio=0.30,
                min_light_area=160,
            ):
                continue
            debug_stats["adjacent_dark_bubble_candidates"] += 1
            before = len(recovered_page["texts"])
            output_page_space = bool((_coerce_bbox(text.get("bbox")) or [0, 0, 0, 0])[1] >= height)
            if _append_runtime_recovered_lobe(
                text,
                candidate,
                candidate,
                output_page_space=output_page_space,
                trim_leading_edge_fragments=True,
            ):
                for recovered_text in recovered_page["texts"][before:]:
                    if isinstance(recovered_text, dict):
                        flags = list(recovered_text.get("qa_flags") or [])
                        if "adjacent_dark_bubble_reocr" not in flags:
                            flags.append("adjacent_dark_bubble_reocr")
                        recovered_text["qa_flags"] = flags
                debug_stats["adjacent_dark_bubble_recovered"] += len(recovered_page["texts"]) - before
                existing_bboxes.extend(
                    _coerce_bbox(item.get("text_pixel_bbox") or item.get("bbox") or item.get("source_bbox"))
                    for item in recovered_page["texts"][before:]
                    if isinstance(item, dict)
                )

    def _recover_detected_dark_balloons_without_text() -> None:
        existing_bboxes: list[list[int]] = []
        for item in list(ocr_page.get("texts") or []) + list(recovered_page.get("texts") or []):
            if not isinstance(item, dict):
                continue
            bbox = _coerce_bbox(item.get("text_pixel_bbox") or item.get("bbox") or item.get("source_bbox"))
            if bbox is not None:
                existing_bboxes.append(bbox)
        for balloon in list(getattr(band, "balloons", []) or []):
            strip_bbox = getattr(balloon, "strip_bbox", None)
            if strip_bbox is None:
                continue
            candidate = [
                int(getattr(strip_bbox, "x1", 0)),
                int(getattr(strip_bbox, "y1", 0)) - int(getattr(band, "y_top", 0) or 0),
                int(getattr(strip_bbox, "x2", 0)),
                int(getattr(strip_bbox, "y2", 0)) - int(getattr(band, "y_top", 0) or 0),
            ]
            x1, y1, x2, y2 = candidate
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            candidate = [x1, y1, x2, y2]
            candidate_area = _bbox_area(candidate)
            if candidate_area <= 0:
                continue
            covered = False
            for existing_bbox in existing_bboxes:
                overlap = _bbox_intersection_area(candidate, existing_bbox)
                if overlap / float(max(1, min(candidate_area, _bbox_area(existing_bbox)))) >= 0.30:
                    covered = True
                    break
            if covered:
                continue
            evidence = _inner_dark_text_evidence(image, BBox(x1, y1, x2, y2))
            if not (
                _dark_bubble_evidence_supports_lobe_reocr(
                    evidence,
                    min_dark_ratio=0.45,
                    max_bright_ratio=0.24,
                    min_light_area=180,
                )
            ):
                continue
            probe_x1 = max(0, x1 - max(80, int((x2 - x1) * 0.55)))
            probe_y1 = max(0, y1 - max(72, int((y2 - y1) * 1.35)))
            probe_x2 = min(width, x2 + max(80, int((x2 - x1) * 0.55)))
            probe_y2 = min(height, y2 + max(72, int((y2 - y1) * 1.35)))
            detected_bubble_bbox = _detect_dark_oval_bubble_bbox(
                image,
                BBox(probe_x1, probe_y1, probe_x2, probe_y2),
            )
            if detected_bubble_bbox is not None:
                detected_area = _bbox_area(detected_bubble_bbox)
                overlap = _bbox_intersection_area(detected_bubble_bbox, candidate)
                min_overlap_ratio = overlap / float(max(1, min(detected_area, candidate_area)))
                if (
                    detected_area <= 0
                    or detected_area > int(candidate_area * 8.0)
                    or min_overlap_ratio < 0.55
                    or _dark_bubble_bbox_is_only_text_candidate(detected_bubble_bbox, candidate)
                ):
                    detected_bubble_bbox = None
            if detected_bubble_bbox is None:
                if existing_bboxes:
                    expanded_candidate = [
                        max(0, x1 - max(40, int((x2 - x1) * 0.18))),
                        max(0, y1 - max(36, int((y2 - y1) * 0.70))),
                        min(width, x2 + max(48, int((x2 - x1) * 0.22))),
                        min(height, y2 + max(40, int((y2 - y1) * 0.72))),
                    ]
                    bubble_bbox = expanded_candidate if _bbox_area(expanded_candidate) <= int(candidate_area * 8.0) else candidate
                else:
                    bubble_bbox = candidate
            else:
                bubble_bbox = list(detected_bubble_bbox)
            debug_stats["lobe_candidates"] += 1
            debug_stats["lobe_evidence_passed"] += 1
            before_count = len(recovered_page["texts"])
            if _append_runtime_recovered_lobe(
                {"confidence": float(getattr(balloon, "confidence", 0.0) or 0.0)},
                candidate,
                bubble_bbox,
            ):
                for recovered_text in recovered_page["texts"][before_count:]:
                    if isinstance(recovered_text, dict):
                        flags = list(recovered_text.get("qa_flags") or [])
                        if "detected_dark_bubble_without_text_reocr" not in flags:
                            flags.append("detected_dark_bubble_without_text_reocr")
                        recovered_text["qa_flags"] = flags
                        existing_bbox = _coerce_bbox(
                            recovered_text.get("text_pixel_bbox")
                            or recovered_text.get("bbox")
                            or recovered_text.get("source_bbox")
                        )
                        if existing_bbox is not None:
                            existing_bboxes.append(existing_bbox)

    for text in list(ocr_page.get("texts") or []):
        if not isinstance(text, dict):
            continue
        raw_text = str(text.get("text") or text.get("original") or "").strip()
        if not raw_text:
            continue
        try:
            confidence = float(
                text.get("confidence_raw")
                if text.get("confidence_raw") not in (None, "")
                else text.get("confidence", 0.0)
            )
        except Exception:
            confidence = 0.0
        bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
        if bbox is None:
            continue
        bubble_source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if (
            isinstance(text.get("connected_position_bboxes"), list)
            and bubble_source not in {"derived_white_crop_rejected", "rejected_derived_bubble_mask"}
        ):
            _recover_layout_connected_lobes(text, bbox)
        if layout_lobes_only:
            _recover_adjacent_dark_bubbles(text, bbox)
        if layout_lobes_only:
            continue
        if _is_dark_bubble_reocr_text(text):
            debug_stats["dark_texts"] += 1
            _recover_uncovered_dark_lobes(text, bbox)
        if confidence > 0.72:
            continue
        debug_stats["low_confidence_attempts"] += 1
        x1, y1, x2, y2 = bbox
        crop_left, crop_top, crop_right, crop_bottom = _partial_dark_bubble_probe_crop_bbox(
            width=width,
            height=height,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if crop_right <= crop_left or crop_bottom <= crop_top:
            continue
        evidence = _inner_dark_text_evidence(image, BBox(crop_left, crop_top, crop_right, crop_bottom))
        if not (
            _dark_bubble_evidence_supports_lobe_reocr(
                evidence,
                min_dark_ratio=0.32,
                max_bright_ratio=0.24,
                min_light_area=180,
            )
        ):
            continue
        dark_oval_bbox = _detect_dark_oval_bubble_bbox(
            image,
            BBox(crop_left, crop_top, crop_right, crop_bottom),
        )
        strong_dark_bubble_visual = bool(
            float(evidence.get("dark_pixel_ratio") or 0.0) >= 0.70
            and float(evidence.get("bright_pixel_ratio") or 0.0) >= 0.035
            and int(evidence.get("inner_light_area") or 0) >= 600
        )
        if dark_oval_bbox is None and not strong_dark_bubble_visual:
            continue
        bubble_bbox = list(dark_oval_bbox or [crop_left, crop_top, crop_right, crop_bottom])
        crop = image[crop_top:crop_bottom, crop_left:crop_right]
        direct_crop_result = _run_direct_paddle_candidate_crop_reocr(
            crop,
            idioma_origem=idioma_origem,
        )
        for direct_text in list(direct_crop_result.get("texts") or []):
            if isinstance(direct_text, dict):
                direct_text["bubble_mask_source"] = "image_dark_bubble_mask"
                direct_text["block_profile"] = direct_text.get("block_profile") or "dark_bubble"
                direct_text["layout_profile"] = direct_text.get("layout_profile") or "dark_bubble"
        direct_texts, direct_blocks = _map_crop_ocr_page_to_band(
            direct_crop_result,
            band_page=page_dict,
            band_id=band_id,
            balloon_local_bbox=bubble_bbox,
            crop_left=crop_left,
            crop_top=crop_top,
            candidate_index=candidate_index,
        )
        if not direct_texts:
            continue
        direct_texts, direct_blocks = _filter_dark_bubble_reocr_to_balloon(
            direct_texts,
            direct_blocks,
            bubble_bbox=bubble_bbox,
        )
        if not direct_texts:
            continue
        _annotate_dark_bubble_recovery(direct_texts, direct_blocks, bubble_bbox=bubble_bbox)
        recovered_page["texts"].extend(direct_texts)
        recovered_page["_vision_blocks"].extend(direct_blocks)
        candidate_index += 1
    if not layout_lobes_only:
        _recover_detected_dark_balloons_without_text()
    stats = ocr_page.setdefault("_ocr_stats", {})
    if isinstance(stats, dict):
        stats["partial_dark_bubble_recovery"] = dict(debug_stats)
    if not recovered_page["texts"]:
        return 0
    merged = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)
    if merged:
        for recovered_text in list(recovered_page.get("texts") or []):
            if not isinstance(recovered_text, dict):
                continue
            for existing_text in list(ocr_page.get("texts") or []):
                if not isinstance(existing_text, dict) or existing_text is recovered_text:
                    continue
                _strip_false_trailing_dark_lobe_token(existing_text, recovered_text)
    if isinstance(stats, dict):
        stats["partial_dark_bubble_recovery"]["merged"] = int(merged)
    return merged


def _run_translate_stage(
    ocr_page: dict,
    *,
    translator,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
) -> BandStageOutput:
    translated_pages = translator.translate_pages(
        [ocr_page],
        obra=obra,
        context=context or {},
        glossario=glossario or {},
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
        models_dir=models_dir,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        translation_context=translation_context,
    )
    translated_page = translated_pages[0] if translated_pages else {"texts": []}
    return BandStageOutput(
        "translate",
        _merge_translated_page_metadata(ocr_page, translated_page),
    )


def _finalize_ocr_page_before_translation(
    ocr_page: dict,
    image_shape: tuple[int, int, int],
    *,
    page_number: int | None,
    total_pages: int | None = None,
    source_language: str = "en",
) -> int:
    if not isinstance(ocr_page, dict):
        return 0
    texts = [text for text in list(ocr_page.get("texts") or []) if isinstance(text, dict)]
    if not texts:
        return 0
    blocks = [block for block in list(ocr_page.get("_vision_blocks") or []) if isinstance(block, dict)]
    before = [
        (
            str(text.get("id") or text.get("text_id") or ""),
            str(text.get("text") or ""),
            str(text.get("translated") or text.get("traduzido") or ""),
        )
        for text in texts
    ]
    try:
        from vision_stack.runtime import _finalize_page_ocr_texts

        final_texts, final_blocks = _finalize_page_ocr_texts(
            texts,
            blocks,
            image_shape,
            page_number=page_number,
            total_pages=total_pages,
            source_language=source_language,
        )
    except Exception:
        return 0
    after = [
        (
            str(text.get("id") or text.get("text_id") or ""),
            str(text.get("text") or ""),
            str(text.get("translated") or text.get("traduzido") or ""),
        )
        for text in final_texts
        if isinstance(text, dict)
    ]
    if before == after and len(final_texts) == len(texts) and len(final_blocks) == len(blocks):
        return 0
    ocr_page["texts"] = final_texts
    ocr_page["_vision_blocks"] = final_blocks
    return 1


def _ensure_text_balloon_bboxes(page: dict, band: Band) -> None:
    height = int(band.strip_slice.shape[0])
    width = int(band.strip_slice.shape[1])
    band_id = str(page.get("_band_id") or "").strip()
    if not band_id:
        try:
            source_page_number = int(page.get("_source_page_number") or page.get("numero") or 1)
            band_index = max(0, int(page.get("_band_index") or 1) - 1)
            band_id = _band_id_for(source_page_number, band_index)
        except Exception:
            band_id = "page_001_band_001"

    balloon_blocks: list[dict] = []
    for bubble_index, balloon in enumerate(list(band.balloons or []), start=1):
        bbox_local = [
            int(balloon.strip_bbox.x1),
            int(balloon.strip_bbox.y1) - int(band.y_top),
            int(balloon.strip_bbox.x2),
            int(balloon.strip_bbox.y2) - int(band.y_top),
        ]
        bbox_local = [
            max(0, min(width, bbox_local[0])),
            max(0, min(height, bbox_local[1])),
            max(0, min(width, bbox_local[2])),
            max(0, min(height, bbox_local[3])),
        ]
        if bbox_local[2] <= bbox_local[0] or bbox_local[3] <= bbox_local[1]:
            continue
        block = {
            "bbox": bbox_local,
            "confidence": float(getattr(balloon, "confidence", 0.0) or 0.0),
            "band_id": band_id,
            "bubble_id": f"{band_id}_bubble_{bubble_index:03d}",
            "bubble_mask_bbox": list(bbox_local),
        }
        bubble_inner_bbox = _inner_visual_rect_bbox(bbox_local, width=width, height=height)
        if bubble_inner_bbox is not None:
            block["bubble_inner_bbox"] = bubble_inner_bbox
        _attach_real_bubble_mask_to_block(block, band.strip_slice)
        balloon_blocks.append(block)

    vision_blocks = balloon_blocks + [vb for vb in page.get("_vision_blocks", []) if isinstance(vb, dict)]
    for txt in page.get("texts", []):
        if _apply_text_rect_fallback_mask(txt, width, height):
            continue
        initial_text_mask_revalidated = False
        txt_source = str(txt.get("bubble_mask_source") or txt.get("bubbleMaskSource") or "").strip().lower()
        if _contract_value_present(txt.get("bubble_mask")) and txt_source in WEAK_BUBBLE_MASK_SOURCES:
            _attach_real_bubble_mask_to_block(txt, band.strip_slice)
            initial_text_mask_revalidated = bool(
                _contract_value_present(txt.get("bubble_mask"))
                and _contract_value_missing(txt.get("bubble_mask_error"))
            )
        tx1, ty1, tx2, ty2 = txt.get("bbox", [0, 0, 0, 0])
        best = None
        best_iou = 0.0
        best_area = 0
        best_mask_applied = initial_text_mask_revalidated
        had_explicit_balloon_bbox = _contract_value_present(txt.get("balloon_bbox"))
        for vb in vision_blocks:
            vb_bbox = _coerce_bbox(vb.get("bbox"))
            if vb_bbox is None:
                continue
            vx1, vy1, vx2, vy2 = vb_bbox
            ix = max(0, min(tx2, vx2) - max(tx1, vx1))
            iy = max(0, min(ty2, vy2) - max(ty1, vy1))
            inter = ix * iy
            ta = max(1, (tx2 - tx1) * (ty2 - ty1))
            ratio = inter / ta
            candidate_area = _bbox_area(vb_bbox)
            if ratio > best_iou or (abs(ratio - best_iou) <= 1e-6 and candidate_area > best_area):
                best_iou = ratio
                best = vb
                best_area = candidate_area
        if best:
            best_bbox = _coerce_bbox(best.get("bbox"))
            if _contract_value_missing(txt.get("balloon_bbox")):
                txt_bbox = _coerce_bbox(txt.get("bbox") or txt.get("text_pixel_bbox") or txt.get("source_bbox"))
                txt["balloon_bbox"] = copy.deepcopy(txt_bbox or best_bbox)
            for support_key in (
                "_raw_text_evidence_bbox",
                "_raw_text_evidence_pixels",
                "text_pixel_bbox",
                "layout_bbox",
                "line_polygons",
            ):
                if _contract_value_present(txt.get(support_key)):
                    best[support_key] = copy.deepcopy(txt[support_key])
            _attach_real_bubble_mask_to_block(best, band.strip_slice)
            for metadata_key in ("_raw_text_evidence_bbox", "_raw_text_evidence_pixels", "validated_by_segment_mask"):
                if _contract_value_present(best.get(metadata_key)) and _contract_value_missing(txt.get(metadata_key)):
                    txt[metadata_key] = copy.deepcopy(best[metadata_key])
            best_bubble_mask_bbox = _coerce_bbox(best.get("bubble_mask_bbox") or best_bbox)
            txt_source = str(txt.get("bubble_mask_source") or txt.get("bubbleMaskSource") or "").strip().lower()
            txt_flags = {str(flag).strip() for flag in txt.get("qa_flags") or [] if str(flag).strip()}
            keep_recovered_dark_bubble_contract = (
                txt_source == "image_dark_bubble_mask"
                and (
                    "partial_dark_bubble_lobe_reocr" in txt_flags
                    or "detected_dark_bubble_without_text_reocr" in txt_flags
                    or "candidate_crop_direct_paddle_reocr" in txt_flags
                )
                and _coerce_bbox(txt.get("balloon_bbox")) is not None
            )
            replace_bubble_contract = (
                False
                if keep_recovered_dark_bubble_contract
                else _should_replace_tight_bubble_bbox(txt, best_bubble_mask_bbox)
            )
            best_has_mask = _contract_value_present(best.get("bubble_mask"))
            stale_fallback_source = str(txt.get("bubble_mask_source") or "").strip().lower() in {
                "bbox_fallback",
                "balloon_bbox_fallback",
            }
            if (
                replace_bubble_contract
                or _has_rejected_bubble_mask_contract(txt)
                or (best_has_mask and stale_fallback_source)
            ) and best_bubble_mask_bbox is not None:
                txt["bubble_mask_bbox"] = copy.deepcopy(best_bubble_mask_bbox)
                if best_has_mask:
                    txt["bubble_mask"] = copy.deepcopy(best["bubble_mask"])
                    best_mask_applied = True
            for key in ("bubble_id", "bubble_inner_bbox", "bubble_mask", "bubble_mask_source", "bubble_mask_error"):
                should_replace_rejected = _has_rejected_bubble_mask_contract(txt) and key in {
                    "bubble_mask",
                    "bubble_mask_source",
                    "bubble_mask_error",
                }
                should_replace_stale_fallback = best_has_mask and stale_fallback_source and key in {
                    "bubble_mask",
                    "bubble_mask_source",
                    "bubble_mask_error",
                }
                if _contract_value_present(best.get(key)) and (
                    _contract_value_missing(txt.get(key)) or should_replace_rejected or should_replace_stale_fallback
                ):
                    if key == "bubble_mask" and best_bubble_mask_bbox is not None:
                        txt["bubble_mask_bbox"] = copy.deepcopy(best_bubble_mask_bbox)
                    txt[key] = copy.deepcopy(best[key])
                    if key == "bubble_mask":
                        best_mask_applied = True
            if best_has_mask and _contract_value_missing(best.get("bubble_mask_error")):
                txt.pop("bubble_mask_error", None)
        else:
            if not txt.get("balloon_bbox"):
                w = page.get("width", band.strip_slice.shape[1])
                h = page.get("height", band.strip_slice.shape[0])
                txt["balloon_bbox"] = [
                    max(0, tx1 - 8), max(0, ty1 - 8),
                    min(w, tx2 + 8), min(h, ty2 + 8),
                ]
        layout_bubble_bbox = _coerce_bbox(txt.get("balloon_bbox"))
        replace_tight_layout_bbox = _should_replace_tight_bubble_bbox(txt, layout_bubble_bbox)
        replace_overlarge_layout_bbox = _should_replace_overlarge_derived_bubble_bbox(txt, layout_bubble_bbox)
        current_bubble_bbox = _coerce_bbox(txt.get("bubble_mask_bbox"))
        current_bubble_source = str(txt.get("bubble_mask_source") or "").strip().lower()
        current_layout_recoverable_source = current_bubble_source in {
            "",
            "bbox_fallback",
            "balloon_bbox_fallback",
            "derived_white_crop_rejected",
        }
        layout_bound_to_existing_bubble = bool(
            txt.get("bubble_id")
            and isinstance(best, dict)
            and txt.get("bubble_id") == best.get("bubble_id")
            and layout_bubble_bbox is not None
            and current_bubble_bbox is not None
            and _bbox_area(layout_bubble_bbox) <= int(max(1, _bbox_area(current_bubble_bbox)) * 1.25)
        )
        prefer_explicit_layout_bbox = bool(
            had_explicit_balloon_bbox
            and layout_bubble_bbox is not None
            and current_bubble_bbox is not None
            and current_bubble_bbox != layout_bubble_bbox
            and (current_layout_recoverable_source or layout_bound_to_existing_bubble)
            and _bbox_area(layout_bubble_bbox) >= int(_bbox_area(current_bubble_bbox) * 0.45)
        )
        if layout_bubble_bbox is not None and (
            (not best_mask_applied and (replace_tight_layout_bbox or replace_overlarge_layout_bbox))
            or (
                best_mask_applied
                and not initial_text_mask_revalidated
                and had_explicit_balloon_bbox
                and (replace_overlarge_layout_bbox or prefer_explicit_layout_bbox)
            )
        ):
            layout_block = {
                "bbox": copy.deepcopy(layout_bubble_bbox),
                "bubble_mask_bbox": copy.deepcopy(layout_bubble_bbox),
                "bubble_id": txt.get("bubble_id") or (best.get("bubble_id") if isinstance(best, dict) else None) or f"{band_id}_bubble_layout",
            }
            for support_key in (
                "_raw_text_evidence_bbox",
                "_raw_text_evidence_pixels",
                "text_pixel_bbox",
                "layout_bbox",
                "bbox",
            ):
                support_value = _coerce_bbox(txt.get(support_key))
                if support_key == "_raw_text_evidence_pixels":
                    if _contract_value_present(txt.get(support_key)):
                        layout_block[support_key] = copy.deepcopy(txt[support_key])
                elif support_value is not None:
                    layout_block[support_key] = copy.deepcopy(support_value)
            bubble_inner_bbox = _inner_visual_rect_bbox(layout_bubble_bbox, width=width, height=height)
            if bubble_inner_bbox is not None:
                layout_block["bubble_inner_bbox"] = bubble_inner_bbox
            _attach_real_bubble_mask_to_block(layout_block, band.strip_slice)
            if _contract_value_present(layout_block.get("bubble_mask")):
                txt["bubble_mask_bbox"] = copy.deepcopy(
                    _coerce_bbox(layout_block.get("bubble_mask_bbox")) or layout_bubble_bbox
                )
                txt["bubble_mask"] = copy.deepcopy(layout_block["bubble_mask"])
                txt["bubble_mask_source"] = copy.deepcopy(layout_block.get("bubble_mask_source") or "derived_white_crop")
                if _contract_value_present(layout_block.get("bubble_mask_error")):
                    txt["bubble_mask_error"] = copy.deepcopy(layout_block["bubble_mask_error"])
                else:
                    txt.pop("bubble_mask_error", None)
                if _contract_value_missing(txt.get("bubble_inner_bbox")) and bubble_inner_bbox is not None:
                    txt["bubble_inner_bbox"] = copy.deepcopy(bubble_inner_bbox)
            elif _contract_value_missing(txt.get("bubble_mask_source")):
                txt["bubble_mask_source"] = copy.deepcopy(layout_block.get("bubble_mask_source") or "bbox_fallback")
                if _contract_value_present(layout_block.get("bubble_mask_error")):
                    txt["bubble_mask_error"] = copy.deepcopy(layout_block["bubble_mask_error"])
        if _contract_value_present(txt.get("bubble_mask")) and _contract_value_missing(txt.get("bubble_mask_source")):
            txt["bubble_mask_source"] = IMAGE_WHITE_BUBBLE_MASK_SOURCE
        _promote_dark_rejected_bubble_to_ellipse_contract(
            txt,
            width=width,
            height=height,
            image_bgr=band.strip_slice,
        )
        _reject_unanchored_derived_bubble_contract(txt)


def _shift_text_visual_context_y(text: dict, delta_y: int) -> dict:
    shifted = dict(text)
    for key in (
        "bbox",
        "source_bbox",
        "text_pixel_bbox",
        "balloon_bbox",
        "layout_bbox",
        "bubble_mask_bbox",
        "bubble_inner_bbox",
    ):
        bbox = _shift_bbox_y(shifted.get(key), delta_y)
        if bbox is not None:
            shifted[key] = bbox
    return shifted


def _inner_visual_rect_bbox(outer_bbox: list[int], *, width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [int(v) for v in outer_bbox]
    rect_w = max(1, x2 - x1)
    rect_h = max(1, y2 - y1)
    pad = max(12, int(min(rect_w, rect_h) * 0.07))
    inner = [
        max(0, x1 + pad),
        max(0, y1 + pad),
        min(int(width), x2 - pad),
        min(int(height), y2 - pad),
    ]
    if inner[2] <= inner[0] or inner[3] <= inner[1]:
        return None
    return inner


def _background_luma_from_record(record: dict) -> float | None:
    background = record.get("background_rgb")
    if not isinstance(background, (list, tuple)) or len(background) < 3:
        return None
    try:
        r, g, b = [float(value) for value in background[:3]]
    except Exception:
        return None
    return r * 0.299 + g * 0.587 + b * 0.114


def _promote_dark_rejected_bubble_to_ellipse_contract(
    text: dict,
    *,
    width: int,
    height: int,
    image_bgr: np.ndarray | None = None,
) -> None:
    if not isinstance(text, dict):
        return
    if bool(text.get("card_panel_text_context")):
        return
    source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
    if source not in {"derived_white_crop_rejected", "rejected_derived_bubble_mask", "bbox_fallback", ""}:
        return
    luma = _background_luma_from_record(text)
    profiles = {
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
    }
    if "dark_panel" in profiles or "system_panel" in profiles:
        return
    bubble_bbox = _coerce_bbox(text.get("balloon_bbox") or text.get("bubble_mask_bbox"))
    text_bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("bbox") or text.get("source_bbox"))
    if bubble_bbox is None or text_bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bubble_bbox]
    x1 = max(0, min(int(width), x1))
    x2 = max(0, min(int(width), x2))
    y1 = max(0, min(int(height), y1))
    y2 = max(0, min(int(height), y2))
    if x2 <= x1 or y2 <= y1:
        return
    bubble_bbox = [x1, y1, x2, y2]
    if luma is None and isinstance(image_bgr, np.ndarray) and image_bgr.ndim >= 2:
        try:
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size:
                if roi.ndim == 3:
                    gray = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                else:
                    gray = roi.astype(np.uint8)
                luma = float(np.median(gray))
        except Exception:
            luma = None
    if luma is None or luma > 105.0:
        return
    bubble_w = max(1, x2 - x1)
    bubble_h = max(1, y2 - y1)
    aspect = bubble_w / float(bubble_h)
    if aspect < 1.05 or aspect > 4.2:
        return
    if _bbox_area(bubble_bbox) < int(max(1, _bbox_area(text_bbox)) * 1.25):
        return
    text_w = max(1, int(text_bbox[2]) - int(text_bbox[0]))
    text_h = max(1, int(text_bbox[3]) - int(text_bbox[1]))
    bubble_area = max(1, _bbox_area(bubble_bbox))
    text_area = max(1, _bbox_area(text_bbox))
    if (
        (text_h <= 20 or (text_h <= 28 and text_w <= 220))
        and (
            bubble_area >= text_area * 18
            or bubble_h >= text_h * 8
            or bubble_w >= text_w * 3.2
        )
    ):
        flags = list(text.get("qa_flags") or [])
        if "dark_bubble_visual_mask_rejected_tiny_text" not in flags:
            flags.append("dark_bubble_visual_mask_rejected_tiny_text")
        text["qa_flags"] = flags
        text["bubble_mask_source"] = "text_rect_fallback"
        text["bubble_mask_bbox"] = list(text_bbox)
        text["balloon_bbox"] = list(text_bbox)
        text["block_profile"] = text.get("block_profile") or "standard"
        text["layout_profile"] = text.get("layout_profile") or "standard"
        return
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    text_cx = (text_bbox[0] + text_bbox[2]) / 2.0
    text_cy = (text_bbox[1] + text_bbox[3]) / 2.0
    if not (x1 - 4 <= text_cx <= x2 + 4 and y1 - 4 <= text_cy <= y2 + 4):
        return
    text["bubble_mask_source"] = "image_dark_bubble_mask"
    text["bubble_mask_bbox"] = bubble_bbox
    text["bubble_mask_shape"] = "ellipse"
    text["bubble_mask_ellipse"] = {
        "center": [round(cx, 3), round(cy, 3)],
        "axes": [float(bubble_w), float(bubble_h)],
        "angle": 0.0,
    }
    metrics = text.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics["image_dark_bubble_mask"] = {
            "source": "image_dark_bubble_mask",
            "shape_kind": "ellipse",
            "detection_space": "balloon_bbox_dark_context",
            "mask_bbox": list(bubble_bbox),
            "anchor_bbox": list(text_bbox),
            "ellipse_center": [round(cx, 3), round(cy, 3)],
            "ellipse_axes": [float(bubble_w), float(bubble_h)],
            "ellipse_angle": 0.0,
        }
    text["block_profile"] = text.get("block_profile") or "dark_bubble"
    text["layout_profile"] = text.get("layout_profile") or "dark_bubble"
    flags = list(text.get("qa_flags") or [])
    for flag in ("dark_bubble_promoted_from_rejected_mask", "dark_bubble_ellipse_bbox_mask"):
        if flag not in flags:
            flags.append(flag)
    text["qa_flags"] = flags
    text.pop("bubble_mask_error", None)
    text.pop("bubbleMaskError", None)


def _stage_text_prefers_dark_panel_contract(text: dict) -> bool:
    if not isinstance(text, dict):
        return False
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    connected_dark_bubble_flags = {
        "dark_bubble_connected_lobe_passthrough",
        "dark_bubble_connected_lobes_promoted",
        "partial_dark_bubble_lobe_reocr",
    }
    metrics = text.get("qa_metrics")
    has_dark_bubble_metric = bool(
        isinstance(metrics, dict)
        and isinstance(metrics.get("image_dark_bubble_mask"), dict)
        and (metrics.get("image_dark_bubble_mask") or {}).get("mask_bbox")
    )
    has_connected_lobes = bool(
        text.get("connected_lobe_bboxes")
        or text.get("connected_position_bboxes")
        or text.get("connected_focus_bboxes")
        or len(text.get("balloon_subregions") or []) >= 2
        or str(text.get("connected_balloon_orientation") or "").strip()
    )
    source = str(text.get("bubble_mask_source") or text.get("balloon_mask_source") or "").strip().lower()
    if source == "image_dark_bubble_mask" and bool(flags & connected_dark_bubble_flags) and has_connected_lobes:
        return False
    if bool(text.get("card_panel_text_context")):
        return True
    if (
        "connected_layout_disabled_dark_panel_visual_mask" in flags
        or "dark_panel_rect_from_dark_bubble_bbox" in flags
        or "dark_panel_rect_from_border_lines" in flags
        or "dark_panel_rect_inferred_before_dark_ellipse" in flags
    ):
        return True
    rejected = metrics.get("image_dark_panel_mask_rejected") if isinstance(metrics, dict) else None
    if isinstance(rejected, list):
        for item in rejected:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or "").strip().lower()
            if reason == "overbroad_against_balloon_bbox":
                return True
    profiles = {
        str(text.get("block_profile") or "").strip().lower(),
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("background_type") or "").strip().lower(),
    }
    return bool(profiles & {"dark_panel", "system_panel", "status_panel", "colored_status_panel"})


def _stage_dark_panel_reference_bbox(text: dict) -> list[int] | None:
    metrics = text.get("qa_metrics")
    rejected = metrics.get("image_dark_panel_mask_rejected") if isinstance(metrics, dict) else None
    if not isinstance(rejected, list):
        return None
    for item in rejected:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip().lower()
        if reason != "overbroad_against_balloon_bbox":
            continue
        for key in ("reference_bbox", "balloon_bbox", "mask_bbox"):
            bbox = _coerce_bbox(item.get(key))
            if bbox is not None:
                return bbox
    return None


def _stage_dark_panel_full_bbox_candidate(text: dict, anchor_bbox: list[int]) -> list[int] | None:
    anchor_area = max(1, _bbox_area(anchor_bbox))
    anchor_w = max(1, int(anchor_bbox[2]) - int(anchor_bbox[0]))
    anchor_h = max(1, int(anchor_bbox[3]) - int(anchor_bbox[1]))
    candidates = [
        _stage_dark_panel_reference_bbox(text),
        _coerce_bbox(text.get("balloon_bbox")),
        _coerce_bbox(text.get("bubble_mask_bbox")),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_area = max(1, _bbox_area(candidate))
        if _bbox_intersection_area(candidate, anchor_bbox) < int(anchor_area * 0.45):
            continue
        if candidate_area > max(anchor_area * 12, anchor_area + 120000):
            continue
        candidate_w = max(1, int(candidate[2]) - int(candidate[0]))
        candidate_h = max(1, int(candidate[3]) - int(candidate[1]))
        text_value = str(
            text.get("translated")
            or text.get("traduzido")
            or text.get("text")
            or text.get("original")
            or ""
        ).strip()
        if (
            0 < len(text_value) <= 18
            and candidate_w > max(anchor_w + 80, int(anchor_w * 1.8))
            and candidate_h > max(anchor_h + 36, int(anchor_h * 2.0))
        ):
            flags = list(text.get("qa_flags") or [])
            if "short_dark_text_full_panel_bbox_rejected" not in flags:
                flags.append("short_dark_text_full_panel_bbox_rejected")
            text["qa_flags"] = flags
            metrics = text.setdefault("qa_metrics", {})
            if isinstance(metrics, dict):
                metrics["short_dark_text_full_panel_bbox_rejected"] = {
                    "candidate_bbox": list(candidate),
                    "anchor_bbox": list(anchor_bbox),
                    "text_length": len(text_value),
                }
            continue
        return candidate
    return None


def _stage_dark_panel_visual_rect_candidate(
    image: np.ndarray | None,
    panel_bbox: list[int] | None,
    anchor_bbox: list[int] | None,
) -> list[int] | None:
    if not isinstance(image, np.ndarray) or image.size == 0 or panel_bbox is None or anchor_bbox is None:
        return None
    panel_area = max(1, _bbox_area(panel_bbox))
    anchor_area = max(1, _bbox_area(anchor_bbox))
    if panel_area <= anchor_area:
        return None
    try:
        px1, py1, px2, py2 = [int(v) for v in panel_bbox]
        pad_x = max(12, int(round((px2 - px1) * 0.10)))
        pad_y = max(10, int(round((py2 - py1) * 0.12)))
        height, width = image.shape[:2]
        probe = BBox(
            max(0, px1 - pad_x),
            max(0, py1 - pad_y),
            min(width, px2 + pad_x),
            min(height, py2 + pad_y),
        )
        detected = _detect_dark_rect_panel_frame_bbox(image, probe)
    except Exception:
        detected = None
    if detected is None:
        return None
    detected_area = max(1, _bbox_area(detected))
    if detected_area <= anchor_area:
        return None
    if _bbox_intersection_area(detected, anchor_bbox) < int(anchor_area * 0.55):
        return None
    if detected_area > panel_area * 0.92:
        return None
    return detected


def _apply_stage_dark_panel_contract(
    text: dict,
    *,
    width: int,
    height: int,
    image: np.ndarray | None = None,
) -> bool:
    if not _stage_text_prefers_dark_panel_contract(text):
        return False
    try:
        band_y_top = int(text.get("_band_y_top") or text.get("band_y_top") or 0)
    except Exception:
        band_y_top = 0

    def _to_stage_bbox(value: object) -> list[int] | None:
        bbox = _coerce_bbox(value)
        if bbox is None:
            return None
        if band_y_top and bbox[1] >= height and bbox[3] > height:
            bbox = [bbox[0], bbox[1] - band_y_top, bbox[2], bbox[3] - band_y_top]
        return bbox

    raw_anchor_bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("source_bbox") or text.get("bbox"))
    anchor_bbox = _to_stage_bbox(raw_anchor_bbox)
    panel_bbox = (
        _stage_dark_panel_full_bbox_candidate(text, raw_anchor_bbox)
        if raw_anchor_bbox is not None
        else None
    )
    full_panel_candidate = _to_stage_bbox(panel_bbox)
    panel_bbox = _to_stage_bbox(panel_bbox)
    visual_rect_candidate = _stage_dark_panel_visual_rect_candidate(image, panel_bbox, anchor_bbox)
    if visual_rect_candidate is not None:
        panel_bbox = visual_rect_candidate
        full_panel_candidate = visual_rect_candidate
        flags = list(text.get("qa_flags") or [])
        if "dark_panel_visual_rect_candidate_selected" not in flags:
            flags.append("dark_panel_visual_rect_candidate_selected")
        text["qa_flags"] = flags
    if panel_bbox is None:
        panel_bbox = _to_stage_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox") or text.get("text_pixel_bbox") or text.get("bbox"))
    if panel_bbox is None or anchor_bbox is None:
        return False
    x1, y1, x2, y2 = [int(v) for v in panel_bbox]
    x1 = max(0, min(int(width), x1))
    x2 = max(0, min(int(width), x2))
    y1 = max(0, min(int(height), y1))
    y2 = max(0, min(int(height), y2))
    if x2 <= x1 or y2 <= y1:
        return False
    panel_bbox = [x1, y1, x2, y2]
    if _bbox_intersection_area(panel_bbox, anchor_bbox) < int(max(1, _bbox_area(anchor_bbox)) * 0.45):
        return False
    full_panel_selected = bool(full_panel_candidate == panel_bbox)
    if not full_panel_selected:
        px1, py1, px2, py2 = panel_bbox
        ax1, ay1, ax2, ay2 = anchor_bbox
        cx = (ax1 + ax2) / 2.0
        cy = (ay1 + ay2) / 2.0
        text_half_w = max(1.0, (ax2 - ax1) / 2.0)
        text_half_h = max(1.0, (ay2 - ay1) / 2.0)
        current_half_w = max(abs(cx - px1), abs(px2 - cx))
        current_half_h = max(abs(cy - py1), abs(py2 - cy))
        max_half_w = max(float(DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER), text_half_w + 8.0)
        max_half_h = max(float(DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER), text_half_h + 8.0)
        if "short_dark_text_full_panel_bbox_rejected" in (text.get("qa_flags") or []):
            max_half_w = min(max_half_w, max(text_half_w + 20.0, 80.0))
            max_half_h = min(max_half_h, max(text_half_h + 14.0, 36.0))
        half_w = min(current_half_w, max_half_w)
        half_h = min(current_half_h, max_half_h)
        panel_bbox = [
            max(0, min(int(width), int(math.floor(cx - half_w)))),
            max(0, min(int(height), int(math.floor(cy - half_h)))),
            max(0, min(int(width), int(math.ceil(cx + half_w)))),
            max(0, min(int(height), int(math.ceil(cy + half_h)))),
        ]
    if panel_bbox[2] <= panel_bbox[0] or panel_bbox[3] <= panel_bbox[1]:
        return False
    text["bubble_mask_source"] = "image_dark_panel_mask"
    text["bubble_mask_bbox"] = panel_bbox
    text["balloon_bbox"] = list(panel_bbox)
    bubble_inner_bbox = _inner_visual_rect_bbox(panel_bbox, width=width, height=height)
    if bubble_inner_bbox is not None:
        text["bubble_inner_bbox"] = bubble_inner_bbox
    text["block_profile"] = "dark_panel"
    text["layout_profile"] = "dark_panel"
    text.pop("bubble_mask_shape", None)
    text.pop("bubble_mask_ellipse", None)
    text.pop("bubbleMaskEllipse", None)
    metrics = text.setdefault("qa_metrics", {})
    if isinstance(metrics, dict):
        metrics.pop("image_dark_bubble_mask", None)
        metrics["image_dark_panel_mask"] = {
            "source": "image_dark_panel_mask",
                "detection_space": "stage_dark_panel_contract",
                "mask_bbox": list(panel_bbox),
                "anchor_bbox": list(anchor_bbox),
                "centered_on_text": not full_panel_selected,
                "full_panel_bbox_selected": full_panel_selected,
                "visual_rect_candidate_selected": visual_rect_candidate is not None,
                "max_half_width_from_text_center": DARK_PANEL_RECT_MAX_HALF_WIDTH_FROM_TEXT_CENTER,
                "max_half_height_from_text_center": DARK_PANEL_RECT_MAX_HALF_HEIGHT_FROM_TEXT_CENTER,
            }
    text["qa_flags"] = [
        flag
        for flag in list(text.get("qa_flags") or [])
        if str(flag) not in {
            "dark_bubble_promoted_from_rejected_mask",
            "dark_bubble_ellipse_bbox_mask",
            "dark_panel_full_bbox_selected",
        }
    ]
    if full_panel_selected:
        flags = list(text.get("qa_flags") or [])
        if "dark_panel_full_bbox_selected" not in flags:
            flags.append("dark_panel_full_bbox_selected")
        text["qa_flags"] = flags
    text.pop("bubble_mask_error", None)
    text.pop("bubbleMaskError", None)
    return True


def _normalize_dark_bubble_contracts_for_stage(page: dict, band_image: np.ndarray | None) -> None:
    if not isinstance(page, dict):
        return
    if isinstance(band_image, np.ndarray) and band_image.ndim >= 2:
        height, width = band_image.shape[:2]
    else:
        width = int(page.get("width") or 0)
        height = int(page.get("height") or 0)
    if width <= 0 or height <= 0:
        return
    for text in list(page.get("texts") or []):
        if not isinstance(text, dict):
            continue
        if "_band_y_top" not in text and "band_y_top" not in text and page.get("_band_y_top") is not None:
            try:
                text["_band_y_top"] = int(page.get("_band_y_top") or 0)
            except Exception:
                pass
        if _apply_stage_dark_panel_contract(text, width=width, height=height, image=band_image):
            continue
        initial_source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if initial_source in {"derived_white_crop_rejected", "rejected_derived_bubble_mask", "bbox_fallback", ""}:
            initial_bubble = _coerce_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"))
            initial_anchor = _coerce_bbox(text.get("text_pixel_bbox") or text.get("source_bbox") or text.get("bbox"))
            if initial_bubble is not None and initial_anchor is not None:
                refined = _refine_stage_dark_bubble_bbox_from_image(
                    text,
                    band_image,
                    bubble_bbox=initial_bubble,
                    anchor_bbox=initial_anchor,
                    width=width,
                    height=height,
                )
                if refined is not None:
                    text["bubble_mask_source"] = "image_dark_bubble_mask"
                    text["bubble_mask_bbox"] = list(refined)
                    text["balloon_bbox"] = list(refined)
                    text["block_profile"] = text.get("block_profile") or "dark_bubble"
                    text["layout_profile"] = text.get("layout_profile") or "dark_bubble"
                    text.pop("bubble_mask_error", None)
                    text.pop("bubbleMaskError", None)
        _promote_dark_rejected_bubble_to_ellipse_contract(
            text,
            width=width,
            height=height,
            image_bgr=band_image,
        )
        if _apply_stage_dark_panel_contract(text, width=width, height=height):
            continue
        source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if source != "image_dark_bubble_mask":
            continue
        bubble_bbox = _coerce_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"))
        anchor_bbox = _coerce_bbox(text.get("text_pixel_bbox") or text.get("source_bbox") or text.get("bbox"))
        if bubble_bbox is None or anchor_bbox is None:
            continue
        compact_ellipse_bbox = _compact_dark_bubble_ellipse_bbox_for_stage(text, bubble_bbox, anchor_bbox)
        if compact_ellipse_bbox is not None:
            bubble_bbox = compact_ellipse_bbox
            text["bubble_mask_bbox"] = list(compact_ellipse_bbox)
            text["balloon_bbox"] = list(compact_ellipse_bbox)
            flags = list(text.get("qa_flags") or [])
            if "dark_bubble_compact_ellipse_bbox_preferred" not in flags:
                flags.append("dark_bubble_compact_ellipse_bbox_preferred")
            text["qa_flags"] = flags
        refined_bbox = _refine_stage_dark_bubble_bbox_from_image(
            text,
            band_image,
            bubble_bbox=bubble_bbox,
            anchor_bbox=anchor_bbox,
            width=width,
            height=height,
        )
        if refined_bbox is not None:
            bubble_bbox = refined_bbox
            text["bubble_mask_bbox"] = list(refined_bbox)
            text["balloon_bbox"] = list(refined_bbox)
        overlap = _bbox_intersection_area(bubble_bbox, anchor_bbox) / float(max(1, _bbox_area(anchor_bbox)))
        if overlap < 0.55:
            continue
        if _bbox_area(bubble_bbox) >= max(_bbox_area(anchor_bbox) * 1.35, _bbox_area(text.get("balloon_bbox") or anchor_bbox) * 1.10):
            text["balloon_bbox"] = list(bubble_bbox)
        x1, y1, x2, y2 = [int(v) for v in bubble_bbox]
        bubble_w = max(1, x2 - x1)
        bubble_h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        text["bubble_mask_shape"] = "ellipse"
        text["bubble_mask_ellipse"] = {
            "center": [round(cx, 3), round(cy, 3)],
            "axes": [float(bubble_w), float(bubble_h)],
            "angle": 0.0,
        }
        text["block_profile"] = text.get("block_profile") or "dark_bubble"
        text["layout_profile"] = text.get("layout_profile") or "dark_bubble"
        metrics = text.setdefault("qa_metrics", {})
        if isinstance(metrics, dict):
            metrics["image_dark_bubble_mask"] = {
                "source": "image_dark_bubble_mask",
                "shape_kind": "ellipse",
                "detection_space": "stage_dark_bubble_contract",
                "mask_bbox": list(bubble_bbox),
                "anchor_bbox": list(anchor_bbox),
                "ellipse_center": [round(cx, 3), round(cy, 3)],
                "ellipse_axes": [float(bubble_w), float(bubble_h)],
                "ellipse_angle": 0.0,
            }
        flags = list(text.get("qa_flags") or [])
        if "dark_bubble_ellipse_bbox_mask" not in flags:
            flags.append("dark_bubble_ellipse_bbox_mask")
        text["qa_flags"] = flags
        text.pop("bubble_mask_error", None)
        text.pop("bubbleMaskError", None)


def _compact_dark_bubble_ellipse_bbox_for_stage(
    text: dict,
    bubble_bbox: list[int],
    anchor_bbox: list[int],
) -> list[int] | None:
    compact_text = re.sub(
        r"\s+",
        "",
        str(text.get("translated") or text.get("traduzido") or text.get("text") or ""),
    )
    word_count = len(str(text.get("translated") or text.get("traduzido") or text.get("text") or "").split())
    if len(compact_text) >= 34 or word_count >= 6:
        return None
    ellipse = text.get("bubble_mask_ellipse")
    if not isinstance(ellipse, dict):
        return None
    center = ellipse.get("center")
    axes = ellipse.get("axes")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    if not isinstance(axes, (list, tuple)) or len(axes) < 2:
        return None
    try:
        cx = float(center[0])
        cy = float(center[1])
        aw = abs(float(axes[0]))
        ah = abs(float(axes[1]))
    except Exception:
        return None
    if aw < 80 or ah < 50:
        return None
    bx1, by1, bx2, by2 = [int(v) for v in bubble_bbox]
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    if aw >= bw * 0.82 and ah >= bh * 0.82:
        return None
    if 0 <= cx <= bw and 0 <= cy <= bh:
        cx += bx1
        cy += by1
    ellipse_bbox = [
        max(0, int(round(cx - aw / 2.0))),
        max(0, int(round(cy - ah / 2.0))),
        int(round(cx + aw / 2.0)),
        int(round(cy + ah / 2.0)),
    ]
    ix1 = max(int(ellipse_bbox[0]), int(bubble_bbox[0]))
    iy1 = max(int(ellipse_bbox[1]), int(bubble_bbox[1]))
    ix2 = min(int(ellipse_bbox[2]), int(bubble_bbox[2]))
    iy2 = min(int(ellipse_bbox[3]), int(bubble_bbox[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    ellipse_bbox = [ix1, iy1, ix2, iy2]
    anchor_area = max(1, _bbox_area(anchor_bbox))
    if _bbox_intersection_area(ellipse_bbox, anchor_bbox) / float(anchor_area) < 0.50:
        return None
    return ellipse_bbox


def _refine_stage_dark_bubble_bbox_from_image(
    text: dict,
    band_image: np.ndarray | None,
    *,
    bubble_bbox: list[int],
    anchor_bbox: list[int],
    width: int,
    height: int,
) -> list[int] | None:
    if not isinstance(band_image, np.ndarray) or band_image.ndim < 2:
        return None
    current_area = max(1, _bbox_area(bubble_bbox))
    anchor_area = max(1, _bbox_area(anchor_bbox))
    current_tight = current_area <= max(anchor_area * 2.8, anchor_area + 14000)
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if not current_tight and "dark_bubble_promoted_from_rejected_mask" not in flags:
        return None
    ax1, ay1, ax2, ay2 = [int(v) for v in anchor_bbox]
    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    pad_x = max(90, int(round(aw * 0.35)))
    pad_top = max(170, int(round(ah * 2.8)))
    pad_bottom = max(80, int(round(ah * 1.65)))
    probe = BBox(
        max(0, ax1 - pad_x),
        max(0, ay1 - pad_top),
        min(width, ax2 + pad_x),
        min(height, ay2 + pad_bottom),
    )
    detected = _detect_dark_oval_bubble_bbox(band_image, probe)
    detected_bbox = _coerce_bbox(detected)
    if detected_bbox is None:
        return None
    if _bbox_intersection_area(detected_bbox, anchor_bbox) < int(anchor_area * 0.70):
        return None
    detected_area = max(1, _bbox_area(detected_bbox))
    if detected_area < max(current_area * 1.30, anchor_area * 1.35):
        return None
    aspect = (detected_bbox[2] - detected_bbox[0]) / float(max(1, detected_bbox[3] - detected_bbox[1]))
    if aspect < 0.75 or aspect > 4.4:
        return None
    flags_list = list(text.get("qa_flags") or [])
    if "dark_bubble_visual_bbox_refined" not in flags_list:
        flags_list.append("dark_bubble_visual_bbox_refined")
    text["qa_flags"] = flags_list
    return list(detected_bbox)


def _normalized_ocr_text_score(value: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+", str(value or ""))
    return sum(len(token) for token in tokens)


def _dark_bubble_full_crop_ocr_is_better(old_text: str, new_text: str) -> bool:
    if TEXT_RECT_FALLBACK_NOTE_PATTERN.match(str(new_text or "")) or TEXT_RECT_FALLBACK_NOTE_PATTERN.match(str(old_text or "")):
        return False
    old_score = _normalized_ocr_text_score(old_text)
    new_score = _normalized_ocr_text_score(new_text)
    if new_score < 8:
        return False
    old_compact = re.sub(r"[^A-Za-z0-9]+", "", str(old_text or "")).lower()
    new_compact = re.sub(r"[^A-Za-z0-9]+", "", str(new_text or "")).lower()
    if (
        old_compact
        and new_compact.startswith(old_compact)
        and len(re.findall(r"[A-Za-z0-9]+", str(old_text or ""))) >= 6
        and len(new_compact) >= len(old_compact) + 6
    ):
        return False
    if old_compact and old_compact in new_compact:
        prefix_len = new_compact.find(old_compact)
        old_word_count = len(re.findall(r"[A-Za-z0-9]+", str(old_text or "")))
        if prefix_len >= 4 and old_word_count >= 6:
            prefix_raw = str(new_text or "")[: max(0, min(len(str(new_text or "")), prefix_len + 8))]
            prefix_tokens = re.findall(r"[A-Za-z0-9]+", prefix_raw)
            if len(prefix_tokens) <= 2:
                return False
    if old_compact and old_compact in new_compact and new_score >= old_score + 4:
        return True
    if new_score >= max(old_score + 10, int(old_score * 1.18)):
        return True
    old_words = set(re.findall(r"[A-Za-z0-9]+", str(old_text or "").lower()))
    new_words = set(re.findall(r"[A-Za-z0-9]+", str(new_text or "").lower()))
    if not old_words:
        return True
    overlap = len(old_words & new_words) / float(max(1, len(old_words)))
    return bool(overlap >= 0.45 and new_score >= old_score + 4)


def _candidate_geometry_bbox(candidate: dict) -> list[int] | None:
    bbox = _coerce_bbox(
        candidate.get("text_pixel_bbox")
        or candidate.get("source_bbox")
        or candidate.get("bbox")
        or candidate.get("layout_bbox")
    )
    if bbox is not None:
        return bbox
    polygons = candidate.get("line_polygons")
    if not isinstance(polygons, list):
        return None
    xs: list[int] = []
    ys: list[int] = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)):
            continue
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                xs.append(int(round(float(point[0]))))
                ys.append(int(round(float(point[1]))))
            except Exception:
                continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _dark_connected_lobe_full_crop_candidate_crosses_lobe(
    text: dict,
    candidate: dict,
    *,
    bubble_bbox: list[int],
    old_text: str,
) -> bool:
    flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
    if not (
        "dark_bubble_connected_lobes_promoted" in flags
        or "dark_bubble_lobe_mask_bbox_preferred" in flags
        or "dark_bubble_connected_lobe_passthrough" in flags
    ):
        return False
    if _normalized_ocr_text_score(old_text) < 12:
        return False
    candidate_bbox = _candidate_geometry_bbox(candidate)
    if candidate_bbox is None:
        return False
    current_bbox = _candidate_geometry_bbox(text)
    old_compact = re.sub(r"[^A-Za-z0-9]+", "", str(old_text or "")).lower()
    new_text = str(candidate.get("text") or candidate.get("raw_ocr") or candidate.get("original") or "")
    new_compact = re.sub(r"[^A-Za-z0-9]+", "", new_text).lower()
    if current_bbox is not None and old_compact and old_compact in new_compact:
        tx1, ty1, tx2, ty2 = [int(v) for v in current_bbox]
        cx1, cy1, cx2, cy2 = [int(v) for v in candidate_bbox]
        current_w = max(1, tx2 - tx1)
        current_h = max(1, ty2 - ty1)
        expands_current = (
            cx1 < tx1 - max(24, int(round(current_w * 0.12)))
            or cx2 > tx2 + max(36, int(round(current_w * 0.18)))
            or cy1 < ty1 - max(22, int(round(current_h * 0.12)))
            or cy2 > ty2 + max(36, int(round(current_h * 0.28)))
        )
        if expands_current and _bbox_area(candidate_bbox) >= max(_bbox_area(current_bbox) + 2400, int(_bbox_area(current_bbox) * 1.65)):
            return True
    cx1, cy1, cx2, cy2 = [int(v) for v in candidate_bbox]
    bx1, by1, bx2, by2 = [int(v) for v in bubble_bbox]
    candidate_area = max(1, _bbox_area(candidate_bbox))
    overlap = _bbox_intersection_area(candidate_bbox, bubble_bbox) / float(candidate_area)
    lobe_w = max(1, bx2 - bx1)
    lobe_h = max(1, by2 - by1)
    tolerance_x = max(18, int(round(lobe_w * 0.08)))
    tolerance_y = max(14, int(round(lobe_h * 0.08)))
    crosses_edge = (
        cx1 < bx1 - tolerance_x
        or cx2 > bx2 + tolerance_x
        or cy1 < by1 - tolerance_y
        or cy2 > by2 + tolerance_y
    )
    return bool(crosses_edge and overlap < 0.86)


def _clean_dark_bubble_full_crop_ocr_text(value: str) -> str:
    text = str(value or "").strip()
    called_match = re.search(r"([I1lT]?\s*a?m\s+called\b.*)", text, flags=re.IGNORECASE)
    if called_match:
        cleaned = called_match.group(1).strip()
        cleaned = re.sub(r"^[1lT]\s*", "I ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^I\s*a?m\b", "I am", cleaned, flags=re.IGNORECASE)
        return cleaned
    if re.search(r"\b(?:quest|luest)\s+completion\s+criteria\b", text, flags=re.IGNORECASE):
        text = re.sub(r"\bLuest\b", "Quest", text, flags=re.IGNORECASE)
        text = re.sub(r"\bLeve\s+underworld\b", "Level 1 underworld", text, flags=re.IGNORECASE)
        text = re.sub(r"\bLeve[lI]?\s*underworld\b", "Level 1 underworld", text, flags=re.IGNORECASE)
    return text


def _offset_polygon_points(points, dx: int, dy: int):
    if not isinstance(points, (list, tuple)):
        return points
    shifted = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                shifted.append([int(round(float(point[0]))) + dx, int(round(float(point[1]))) + dy])
            except Exception:
                shifted.append(point)
        else:
            shifted.append(point)
    return shifted


def _replace_dark_bubble_text_with_full_crop_ocr(
    page: dict,
    band: Band,
    *,
    runtime,
    idioma_origem: str,
    work_title: str = "",
    work_title_user_provided: bool = False,
) -> int:
    if not isinstance(page, dict) or runtime is None:
        return 0
    image = band.strip_slice
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        return 0
    height, width = image.shape[:2]
    replaced = 0
    for text in list(page.get("texts") or []):
        if not isinstance(text, dict):
            continue
        source = str(text.get("bubble_mask_source") or text.get("bubbleMaskSource") or "").strip().lower()
        if source != "image_dark_bubble_mask":
            continue
        flags = {str(flag).strip().lower() for flag in text.get("qa_flags") or [] if str(flag).strip()}
        if not (
            "dark_bubble_visual_bbox_refined" in flags
            or "dark_bubble_promoted_from_rejected_mask" in flags
            or "dark_bubble_oval_reocr" in flags
        ):
            continue
        bubble_bbox = _coerce_bbox(text.get("bubble_mask_bbox") or text.get("balloon_bbox"))
        if bubble_bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bubble_bbox]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 120 or (y2 - y1) < 60:
            continue
        crop = image[y1:y2, x1:x2]
        crop_page = {
            "numero": page.get("numero", 1),
            "width": int(crop.shape[1]),
            "height": int(crop.shape[0]),
            "_vision_blocks": [
                {
                    "bbox": [0, 0, int(crop.shape[1]), int(crop.shape[0])],
                    "confidence": 1.0,
                    "detector": "dark_bubble_full_crop_reocr",
                }
            ],
        }
        try:
            if work_title or work_title_user_provided:
                try:
                    ocr_result = runtime.run_ocr_stage(
                        crop,
                        crop_page,
                        work_title=work_title,
                        work_title_user_provided=work_title_user_provided,
                    )
                except TypeError as exc:
                    message = str(exc)
                    if "unexpected keyword argument" not in message and "got an unexpected keyword" not in message:
                        raise
                    ocr_result = runtime.run_ocr_stage(crop, crop_page)
            else:
                ocr_result = runtime.run_ocr_stage(crop, crop_page)
        except Exception:
            continue
        candidates = [item for item in list((ocr_result or {}).get("texts") or []) if isinstance(item, dict)]
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                _normalized_ocr_text_score(str(item.get("text") or item.get("raw_ocr") or "")),
                float(item.get("confidence") or 0.0),
            ),
            reverse=True,
        )
        best = candidates[0]
        new_text = _clean_dark_bubble_full_crop_ocr_text(str(best.get("text") or best.get("raw_ocr") or "").strip())
        old_text = str(text.get("text") or text.get("raw_ocr") or text.get("original") or "").strip()
        geometry_probe = copy.deepcopy(best)
        for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
            probe_bbox = _coerce_bbox(geometry_probe.get(key))
            if probe_bbox is not None:
                geometry_probe[key] = [probe_bbox[0] + x1, probe_bbox[1] + y1, probe_bbox[2] + x1, probe_bbox[3] + y1]
        if isinstance(geometry_probe.get("line_polygons"), list):
            geometry_probe["line_polygons"] = [_offset_polygon_points(poly, x1, y1) for poly in geometry_probe.get("line_polygons") or []]
        if _dark_connected_lobe_full_crop_candidate_crosses_lobe(
            text,
            geometry_probe,
            bubble_bbox=bubble_bbox,
            old_text=old_text,
        ):
            flags_list = list(text.get("qa_flags") or [])
            if "dark_bubble_full_crop_reocr_rejected_cross_lobe_geometry" not in flags_list:
                flags_list.append("dark_bubble_full_crop_reocr_rejected_cross_lobe_geometry")
            text["qa_flags"] = flags_list
            continue
        if not _dark_bubble_full_crop_ocr_is_better(old_text, new_text):
            continue
        text["text"] = new_text
        text["raw_ocr"] = new_text
        text["original"] = new_text
        text["source_text"] = new_text
        text["confidence"] = best.get("confidence", text.get("confidence", 1.0))
        for key in ("bbox", "source_bbox", "text_pixel_bbox", "layout_bbox"):
            bbox = _coerce_bbox(best.get(key))
            if bbox is not None:
                text[key] = [bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1]
        if _coerce_bbox(text.get("text_pixel_bbox")) is None:
            text["text_pixel_bbox"] = list(text.get("bbox") or bubble_bbox)
        if _coerce_bbox(text.get("bbox")) is None:
            text["bbox"] = list(text.get("text_pixel_bbox") or bubble_bbox)
        if isinstance(best.get("line_polygons"), list):
            text["line_polygons"] = [_offset_polygon_points(poly, x1, y1) for poly in best.get("line_polygons") or []]
        text["bubble_mask_source"] = "image_dark_bubble_mask"
        text["bubble_mask_bbox"] = list(bubble_bbox)
        text["balloon_bbox"] = list(bubble_bbox)
        text["block_profile"] = text.get("block_profile") or "dark_bubble"
        text["layout_profile"] = text.get("layout_profile") or "dark_bubble"
        flags_list = list(text.get("qa_flags") or [])
        if "dark_bubble_full_crop_reocr_replaced" not in flags_list:
            flags_list.append("dark_bubble_full_crop_reocr_replaced")
        text["qa_flags"] = flags_list
        _strip_false_trailing_dark_bubble_fragment(text)
        replaced += 1
    return replaced


def _recover_top_narration_visual_rect_from_page(
    page: dict,
    *,
    band: Band,
    page_image_bgr: np.ndarray | None,
    page_y_top: int,
) -> None:
    # top_narration foi desativado como perfil interno. A preservacao da moldura
    # agora deve vir das regras genericas de caixa/retangulo no inpaint/typeset.
    return


def _run_review_layout_stage(
    band: Band,
    *,
    ocr_page: dict,
    band_history: list[dict] | None = None,
    connected_reasoner_config: dict | None = None,
    layout_page_image_bgr: np.ndarray | None = None,
    layout_page_y_top: int = 0,
) -> BandStageOutput:
    from ocr.contextual_reviewer import contextual_review_page
    from layout.balloon_layout import enrich_page_layout
    import cv2

    reviewed_page = contextual_review_page(copy.deepcopy(ocr_page), band_history or [], [])
    if connected_reasoner_config:
        reviewed_page["_connected_balloon_reasoner"] = connected_reasoner_config

    reviewed_page["_cached_image_bgr"] = cv2.cvtColor(band.strip_slice, cv2.COLOR_RGB2BGR)
    reviewed_page = enrich_page_layout(reviewed_page)
    _recover_top_narration_visual_rect_from_page(
        reviewed_page,
        band=band,
        page_image_bgr=layout_page_image_bgr,
        page_y_top=layout_page_y_top,
    )
    _ensure_text_balloon_bboxes(reviewed_page, band)
    return BandStageOutput("review_layout", reviewed_page)


def _collect_inpaint_perf_updates(translated_page: dict) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "_strip_fast_solid_balloon_count" in translated_page:
        updates["fast_solid_balloon_count"] = int(translated_page.get("_strip_fast_solid_balloon_count") or 0)
    if "_strip_fast_solid_white_count" in translated_page:
        updates["fast_solid_white_count"] = int(translated_page.get("_strip_fast_solid_white_count") or 0)
    if "_strip_fast_solid_black_count" in translated_page:
        updates["fast_solid_black_count"] = int(translated_page.get("_strip_fast_solid_black_count") or 0)
    if "_strip_fast_solid_colored_count" in translated_page:
        updates["fast_solid_colored_count"] = int(translated_page.get("_strip_fast_solid_colored_count") or 0)
    if "_strip_fast_solid_fill_samples" in translated_page:
        samples = translated_page.get("_strip_fast_solid_fill_samples")
        if isinstance(samples, list):
            updates["fast_solid_fill_samples"] = list(samples)
    if "_strip_fast_white_balloon_count" in translated_page:
        updates["fast_white_balloon_count"] = int(translated_page.get("_strip_fast_white_balloon_count") or 0)
    if "_strip_connected_white_geometry_fill_count" in translated_page:
        updates["connected_white_geometry_fill_count"] = int(
            translated_page.get("_strip_connected_white_geometry_fill_count") or 0
        )
    if "_strip_connected_white_geometry_fill_mask_pixels" in translated_page:
        updates["connected_white_geometry_fill_mask_pixels"] = int(
            translated_page.get("_strip_connected_white_geometry_fill_mask_pixels") or 0
        )
    if "_strip_fast_local_balloon_count" in translated_page:
        updates["fast_local_balloon_count"] = int(translated_page.get("_strip_fast_local_balloon_count") or 0)
    if "_strip_fast_dark_panel_fill_count" in translated_page:
        updates["fast_dark_panel_fill_count"] = int(translated_page.get("_strip_fast_dark_panel_fill_count") or 0)
    if "_strip_dark_panel_fill_count" in translated_page:
        updates["dark_panel_fill_count"] = int(translated_page.get("_strip_dark_panel_fill_count") or 0)
    if "_strip_remaining_inpaint_blocks" in translated_page:
        updates["remaining_inpaint_blocks"] = int(translated_page.get("_strip_remaining_inpaint_blocks") or 0)
    for reason_key, perf_key in (
        ("_strip_fast_solid_rejection_reasons", "fast_solid_rejection_reasons"),
        ("_strip_fast_solid_fill_reject_reasons", "fast_solid_fill_reject_reasons"),
        ("_strip_fast_white_rejection_reasons", "fast_white_rejection_reasons"),
        ("_strip_connected_white_rejection_reasons", "connected_white_rejection_reasons"),
        ("_strip_fast_local_rejection_reasons", "fast_local_rejection_reasons"),
        ("_strip_fast_dark_rejection_reasons", "fast_dark_rejection_reasons"),
    ):
        reasons = translated_page.get(reason_key)
        if isinstance(reasons, dict):
            updates[perf_key] = dict(reasons)
    for flag in (
        "used_fast_white_fill",
        "used_fast_solid_fill",
        "used_fast_dark_fill",
        "used_fast_local_fill",
        "used_real_inpaint",
        "used_post_cleanup",
    ):
        key = f"_strip_{flag}"
        if key in translated_page:
            updates[flag] = bool(translated_page.get(key))
    for key in (
        "_t_roi_select_ms",
        "_t_lama_ms",
        "_t_lama_total_ms",
        "_t_cleanup_total_ms",
        "_t_cleanup_seam_ms",
        "_t_cleanup_band_artifact_ms",
        "_t_cleanup_white_line_ms",
        "_t_cleanup_white_box_ms",
        "_t_cleanup_micro_ms",
        "used_roi_crop",
        "roi_area_ratio",
        "cleanup_reason",
        "cleanup_skipped_seam",
        "cleanup_skipped_band_artifact",
        "cleanup_skipped_white_line",
        "cleanup_skipped_white_box",
    ):
        if key in translated_page:
            updates[key] = translated_page.get(key)
    return updates


def _run_inpaint_stage(
    band: Band,
    *,
    inpainter,
    translated_page: dict,
    band_index: int | None = None,
    source_page_number: int | None = None,
) -> BandImageStageOutput:
    compat_text_fields = _legacy_decision_fields_by_record(translated_page.get("texts"))
    page_for_inpaint = _without_legacy_decision_fields_for_stage(translated_page)
    if band_index is not None:
        page_for_inpaint["_band_index"] = int(band_index)
    elif "_band_index" not in page_for_inpaint:
        page_for_inpaint["_band_index"] = 0
    if source_page_number is not None:
        page_for_inpaint["_source_page_number"] = int(source_page_number)
    elif "_source_page_number" not in page_for_inpaint:
        page_for_inpaint["_source_page_number"] = int(page_for_inpaint.get("numero") or 0)
    page_for_inpaint["_band_y_top"] = int(band.y_top)
    _normalize_dark_bubble_contracts_for_stage(page_for_inpaint, band.strip_slice)
    _drop_suppressed_records_for_inpaint(page_for_inpaint)
    cleaned = inpainter.inpaint_band_image(band.strip_slice, page_for_inpaint)
    for key in ("texts", "_vision_blocks"):
        value = page_for_inpaint.get(key)
        if isinstance(value, list):
            translated_page[key] = copy.deepcopy(value)
    _restore_legacy_decision_fields(translated_page.get("texts"), compat_text_fields)
    translated_page.update(
        {
            key: value
            for key, value in page_for_inpaint.items()
            if str(key).startswith("_strip_")
            or key
            in {
                "used_roi_crop",
                "roi_area_ratio",
                "cleanup_reason",
                "cleanup_skipped_seam",
                "cleanup_skipped_band_artifact",
                "cleanup_skipped_white_line",
                "cleanup_skipped_white_box",
            }
        }
    )
    return BandImageStageOutput(
        "inpaint",
        cleaned,
        _collect_inpaint_perf_updates(translated_page),
    )


def _run_typeset_stage(
    cleaned_slice: np.ndarray,
    *,
    typesetter,
    translated_page: dict,
) -> BandImageStageOutput:
    compat_text_fields = _legacy_decision_fields_by_record(translated_page.get("texts"))
    page_for_typeset = _without_legacy_decision_fields_for_stage(translated_page)
    _normalize_dark_bubble_contracts_for_stage(page_for_typeset, cleaned_slice)
    rendered = typesetter.render_band_image(cleaned_slice, page_for_typeset)
    for key in ("texts", "_vision_blocks", "_bubble_regions"):
        value = page_for_typeset.get(key)
        if isinstance(value, list):
            translated_page[key] = copy.deepcopy(value)
    _restore_legacy_decision_fields(translated_page.get("texts"), compat_text_fields)
    return BandImageStageOutput(
        "typeset",
        rendered,
    )


def _run_copy_back_stage(
    band: Band,
    *,
    cleaned_slice: np.ndarray | None = None,
    rendered_slice: np.ndarray,
    translated_page: dict,
) -> BandImageStageOutput:
    return BandImageStageOutput(
        "copy_back",
        _apply_copy_back_outside_balloons(
            band,
            ocr_page=translated_page,
            rendered_slice=rendered_slice,
            cleaned_slice=cleaned_slice,
        ),
    )


def _commit_band_outputs(
    band: Band,
    *,
    cleaned_slice: np.ndarray,
    rendered_slice: np.ndarray,
    ocr_result: dict,
) -> Band:
    band.cleaned_slice = np.array(cleaned_slice, copy=True)
    band.rendered_slice = np.array(rendered_slice, copy=True)
    band.ocr_result = copy.deepcopy(ocr_result)
    return band


def _all_translations_unchanged(page: dict) -> bool:
    texts = list((page or {}).get("texts") or [])
    if not texts:
        return False
    for text in texts:
        if not isinstance(text, dict):
            return False
        source = str(text.get("original") or text.get("text") or "").strip()
        translated = str(text.get("translated") or text.get("traduzido") or "").strip()
        if not source or not translated or source != translated:
            return False
    return True


def _all_texts_skip_processing(page: dict) -> bool:
    return False


def _scanlation_discord_promo_text_signal(raw: object) -> tuple[bool, bool, bool]:
    text = str(raw or "").strip()
    if not text:
        return False, False, False
    normalized = re.sub(r"[^A-Za-z0-9:/._#-]+", " ", text).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    has_discord = bool(
        re.search(r"\b(?:d|)iscord\.gg\b", normalized)
        or "discordgg" in compact
        or "iscordgg" in compact
        or "discord" in normalized
    )
    has_invite = bool(
        "join" in normalized
        or "support us" in normalized
        or "supportus" in compact
        or "invite" in normalized
    )
    has_url = bool(
        re.search(r"https?://|www\.|(?:d|)iscord\.gg|\.gg\b", normalized)
        or "discordgg" in compact
        or "iscordgg" in compact
    )
    return has_discord, has_invite, has_url


def _ocr_page_is_scanlation_discord_promo_band(page: dict, band: Band) -> bool:
    texts = [text for text in list((page or {}).get("texts") or []) if isinstance(text, dict)]
    if not texts:
        return False
    signals = []
    for text in texts:
        raw = text.get("text") or text.get("original") or text.get("raw_ocr") or text.get("translated") or ""
        signals.append(_scanlation_discord_promo_text_signal(raw))
    has_discord = any(item[0] for item in signals)
    has_invite = any(item[1] for item in signals)
    has_url = any(item[2] for item in signals)
    if not has_discord:
        return False
    all_credit_like = all(
        _candidate_crop_reocr_text_is_scanlation_credit(
            text.get("text") or text.get("original") or text.get("raw_ocr") or text.get("translated") or ""
        )
        or _scanlation_discord_promo_text_signal(
            text.get("text") or text.get("original") or text.get("raw_ocr") or text.get("translated") or ""
        )[2]
        for text in texts
    )
    if not all_credit_like:
        return False
    if has_url and (has_invite or len(texts) <= 2):
        return True
    if has_discord and has_invite and len(texts) <= 6:
        return True
    return False


def _mark_scanlation_discord_promo_band(page: dict, perf: dict) -> dict:
    updated = dict(page or {})
    flags = list(updated.get("qa_flags") or [])
    if "scanlation_discord_promo" not in flags:
        flags.append("scanlation_discord_promo")
    updated["qa_flags"] = flags
    updated["content_class"] = "scanlation_credit"
    updated["export_policy"] = "exclude_from_translated_output"
    updated["translate_policy"] = "skip"
    updated["inpaint_policy"] = "skip"
    updated["render_policy"] = "skip"
    updated["route_action"] = "review_required"
    updated["route_reason"] = "scanlation_discord_promo"
    updated["excluded_non_story"] = True
    updated["exclusion_reason"] = "scanlation_discord_promo"
    updated["review_required"] = False
    for text in list(updated.get("texts") or []):
        if not isinstance(text, dict):
            continue
        text_flags = list(text.get("qa_flags") or [])
        if "scanlation_discord_promo" not in text_flags:
            text_flags.append("scanlation_discord_promo")
        text["qa_flags"] = text_flags
        text["content_class"] = "scanlation_credit"
        text["export_policy"] = "exclude_from_translated_output"
        text["translate_policy"] = "skip"
        text["inpaint_policy"] = "skip"
        text["render_policy"] = "skip"
        text["skip_processing"] = True
        text["skip_reason"] = "scanlation_discord_promo"
        text["route_action"] = "review_required"
        text["route_reason"] = "scanlation_discord_promo"
        text["visible"] = False
    perf["excluded_non_story"] = True
    perf["exclusion_reason"] = "scanlation_discord_promo"
    return updated


def _smart_skip_shadow_enabled() -> bool:
    value = os.environ.get("TRADUZAI_SMART_SKIP_SHADOW", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _smart_skip_real_enabled() -> bool:
    value = os.environ.get("TRADUZAI_SMART_SKIP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_smart_skip_shadow(page: dict, perf: dict) -> None:
    from strip.smart_skip import annotate_page_with_smart_skip_shadow

    annotate_page_with_smart_skip_shadow(page)
    shadow = page.get("_smart_skip_shadow") or {}
    perf["smart_skip_shadow_candidate_count"] = int(shadow.get("candidate_count") or 0)
    perf["smart_skip_shadow_not_safe_count"] = int(shadow.get("not_safe_count") or 0)
    perf["smart_skip_shadow_category_counts"] = dict(shadow.get("category_counts") or {})


def _apply_smart_skip_real(page: dict, perf: dict) -> bool:
    from strip.smart_skip import annotate_page_with_smart_skip_shadow

    annotate_page_with_smart_skip_shadow(page)
    shadow = page.get("_smart_skip_shadow") or {}
    candidates = list(shadow.get("candidates") or [])
    candidate_count = int(shadow.get("candidate_count") or 0)
    not_safe_count = int(shadow.get("not_safe_count") or 0)
    category_counts = dict(shadow.get("category_counts") or {})

    perf["smart_skip_real_candidate_count"] = candidate_count
    perf["smart_skip_real_not_safe_count"] = not_safe_count
    perf["smart_skip_real_category_counts"] = category_counts
    perf["smart_skip_real_applied"] = False

    texts = [text for text in list(page.get("texts") or []) if isinstance(text, dict)]
    if not texts or candidate_count != len(texts) or not_safe_count:
        return False

    candidates_by_index = {
        int(candidate.get("text_index")): candidate
        for candidate in candidates
        if candidate.get("text_index") is not None
    }
    if len(candidates_by_index) != len(texts):
        return False

    for index, text in enumerate(texts):
        decision = dict(candidates_by_index[index])
        text["smart_skip_decision"] = decision
    return False


def process_band(
    band: Band,
    runtime,
    translator,
    inpainter,
    typesetter,
    page_idx: int,
    context: dict | None = None,
    glossario: dict | None = None,
    idioma_origem: str = "en",
    idioma_destino: str = "pt-BR",
    obra: str = "",
    work_title_user_provided: bool = False,
    connected_reasoner_config: dict | None = None,
    band_history: list[dict] | None = None,
    source_page_number: int | None = None,
    models_dir: str = "",
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "traduzai-translator",
    translation_context: dict | None = None,
    precomputed_ocr_page: dict | None = None,
    ordered_context_after_translate_callback=None,
    layout_page_image_bgr: np.ndarray | None = None,
    layout_page_y_top: int = 0,
    gpu_stage_lock=None,
    ocr_stage_lock=None,
    inpaint_stage_lock=None,
    typeset_stage_lock=None,
) -> Band:


    """Processa uma banda pelas stages OCR -> translate -> inpaint -> typeset."""
    total_start = time.perf_counter()
    durations: dict[str, float] = {}
    perf = {
        "band_index": int(page_idx),
        "y_top": int(band.y_top),
        "y_bottom": int(band.y_bottom),
        "height": int(band.height),
        "balloon_count": int(len(band.balloons)),
        "ocr_text_count": 0,
        "text_count": 0,
        "durations_sec": durations,
    }
    band_id = _band_id_for(source_page_number or page_idx + 1, page_idx)

    def _mark(stage: str, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        durations[stage] = round(elapsed, 4)
        perf[f"_t_{stage}_ms"] = round(elapsed * 1000.0, 3)

    def _mark_stage_elapsed(
        stage: str,
        total_elapsed: float,
        *,
        wait_elapsed: float | None = None,
        compute_elapsed: float | None = None,
    ) -> None:
        durations[stage] = round(total_elapsed, 4)
        perf[f"_t_{stage}_ms"] = round(total_elapsed * 1000.0, 3)
        if wait_elapsed is not None:
            wait_elapsed = max(0.0, float(wait_elapsed))
            durations[f"{stage}_wait"] = round(wait_elapsed, 4)
            perf[f"_t_{stage}_wait_ms"] = round(wait_elapsed * 1000.0, 3)
        if compute_elapsed is not None:
            compute_elapsed = max(0.0, float(compute_elapsed))
            durations[f"{stage}_compute"] = round(compute_elapsed, 4)
            perf[f"_t_{stage}_compute_ms"] = round(compute_elapsed * 1000.0, 3)

    def _run_with_stage_lock(stage: str, lock, callback):
        total_started = time.perf_counter()
        context = lock if lock is not None else nullcontext()
        with context:
            compute_started = time.perf_counter()
            output = callback()
            compute_elapsed = time.perf_counter() - compute_started
        total_elapsed = time.perf_counter() - total_started
        _mark_stage_elapsed(
            stage,
            total_elapsed,
            wait_elapsed=total_elapsed - compute_elapsed,
            compute_elapsed=compute_elapsed,
        )
        return output

    def _finish(ocr_result: dict | None = None) -> None:
        if isinstance(ocr_result, dict):
            perf["text_count"] = int(len(ocr_result.get("texts") or []))
            perf["total_sec"] = round(time.perf_counter() - total_start, 4)
            ocr_result["_perf"] = dict(perf)
        else:
            perf["total_sec"] = round(time.perf_counter() - total_start, 4)
        band.perf = dict(perf)
    if not band.balloons:
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result={"texts": [], "_vision_blocks": []},
        )
        _record_copyback_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=band.ocr_result,
            applied=False,
            reason="no_balloons",
        )
        _finish(band.ocr_result)
        return band

    page_dict = _band_to_page_dict(band, page_idx, source_page_number=source_page_number)
    ocr_lock = ocr_stage_lock if ocr_stage_lock is not None else gpu_stage_lock
    ocr_stage = _run_with_stage_lock(
        "ocr",
        ocr_lock,
        lambda: _run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
            work_title=obra,
            work_title_user_provided=work_title_user_provided,
        ),
    )
    ocr_page = ocr_stage.to_page_dict()
    for key in ("numero", "width", "height", "_band_id", "_band_y_top", "_band_index", "_source_page_number"):
        if (key not in ocr_page or ocr_page.get(key) in (None, "")) and key in page_dict:
            ocr_page[key] = page_dict[key]
    band_id = str(page_dict.get("_band_id") or band_id)
    _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
    _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
    perf.update(dict(ocr_stage.perf_updates))
    perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
    ocr_stats = ocr_page.get("_ocr_stats") if isinstance(ocr_page, dict) else None
    if isinstance(ocr_stats, dict):
        for key in (
            "full_page_mapped",
            "crop_fallback_max",
            "sparse_crop_fallback_max",
            "crop_fallback_attempts",
            "crop_fallback_recovered",
            "crop_fallback_suppressed",
        ):
            if key in ocr_stats:
                try:
                    perf[f"ocr_{key}"] = int(ocr_stats.get(key) or 0)
                except Exception:
                    continue
        if "quick_skipped_no_text" in ocr_stats:
            perf["ocr_quick_skipped_no_text"] = bool(ocr_stats.get("quick_skipped_no_text"))
        if "scanlation_credit_skipped" in ocr_stats:
            perf["ocr_scanlation_credit_skipped"] = bool(ocr_stats.get("scanlation_credit_skipped"))
        if "cover_editorial_skipped" in ocr_stats:
            perf["ocr_cover_editorial_skipped"] = bool(ocr_stats.get("cover_editorial_skipped"))
        if "macro_ocr_real" in ocr_stats:
            perf["ocr_macro_ocr_real"] = bool(ocr_stats.get("macro_ocr_real"))
        if "macro_ocr_page_window_owner" in ocr_stats:
            perf["ocr_macro_ocr_page_window_owner"] = bool(
                ocr_stats.get("macro_ocr_page_window_owner")
            )
        for key in (
            "macro_window_count",
            "macro_window_reports",
            "macro_ocr_page_number",
            "macro_ocr_block_count",
            "macro_ocr_empty_record_count",
        ):
            if key in ocr_stats:
                try:
                    perf[f"ocr_{key}"] = int(ocr_stats.get(key) or 0)
                except Exception:
                    continue
        for key in (
            "ocr_cache_hits",
            "ocr_cache_misses",
            "ocr_dedup_removed",
            "quick_text_check_stage",
            "ocr_run_on_suspect_count",
            "ocr_run_on_resolved_count",
        ):
            if key in ocr_stats:
                perf[key] = ocr_stats.get(key)
    elif isinstance(ocr_page, dict) and "quick_skipped_no_text" in ocr_page:
        perf["ocr_quick_skipped_no_text"] = bool(ocr_page.get("quick_skipped_no_text"))
    if isinstance(ocr_page, dict) and "scanlation_credit_skipped" in ocr_page:
        perf["ocr_scanlation_credit_skipped"] = bool(ocr_page.get("scanlation_credit_skipped"))
    if isinstance(ocr_page, dict) and "cover_editorial_skipped" in ocr_page:
        perf["ocr_cover_editorial_skipped"] = bool(ocr_page.get("cover_editorial_skipped"))
    if not list(ocr_page.get("texts") or []):
        recovery_stage = _run_with_stage_lock(
            "ocr_candidate_recovery",
            ocr_lock,
            lambda: _recover_empty_ocr_with_candidate_crops(
                band,
                runtime=runtime,
                page_dict=page_dict,
                band_id=band_id,
                work_title=obra,
                work_title_user_provided=work_title_user_provided,
            ),
        )
        recovered_page = recovery_stage.to_page_dict()
        perf.update(dict(recovery_stage.perf_updates))
        if list(recovered_page.get("texts") or []):
            ocr_page = recovered_page
            for key in ("numero", "width", "height", "_band_id", "_band_y_top", "_band_index", "_source_page_number"):
                if (key not in ocr_page or ocr_page.get(key) in (None, "")) and key in page_dict:
                    ocr_page[key] = page_dict[key]
            _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
            _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
            perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
        else:
            ocr_page = recovered_page
    elif int(perf.get("ocr_text_count") or 0) > 0:
        recovery_stage = _run_with_stage_lock(
            "ocr_candidate_recovery",
            ocr_lock,
            lambda: _recover_empty_ocr_with_candidate_crops(
                band,
                runtime=runtime,
                page_dict=page_dict,
                band_id=band_id,
                work_title=obra,
                work_title_user_provided=work_title_user_provided,
            ),
        )
        recovered_page = recovery_stage.to_page_dict()
        perf.update(dict(recovery_stage.perf_updates))
        merged_recovered = _merge_candidate_crop_recovery_into_ocr_page(ocr_page, recovered_page)
        if merged_recovered > 0:
            for key in ("numero", "width", "height", "_band_id", "_band_y_top", "_band_index", "_source_page_number"):
                if (key not in ocr_page or ocr_page.get(key) in (None, "")) and key in page_dict:
                    ocr_page[key] = page_dict[key]
            _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
            _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
            perf["ocr_candidate_crop_merged_recovered"] = int(merged_recovered)
            perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
    negative_promoted = fuse_negative_dark_bubble_candidates(
        ocr_page,
        ocr_page.get("_negative_evidence") if isinstance(ocr_page, dict) else None,
        band.strip_slice,
    )
    if int(negative_promoted or 0) > 0:
        _normalize_dark_bubble_contracts_for_stage(ocr_page, band.strip_slice)
        _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
        _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
        perf["ocr_negative_dark_promoted"] = int(negative_promoted or 0)
        perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
    partial_dark_recovered = _run_with_stage_lock(
        "ocr_partial_dark_bubble_recovery",
        ocr_lock,
        lambda: _recover_partial_dark_bubble_ocr_from_texts(
            ocr_page,
            band=band,
            page_dict=page_dict,
            band_id=band_id,
            idioma_origem=idioma_origem,
            runtime=runtime,
            work_title=obra,
            work_title_user_provided=work_title_user_provided,
        ),
    )
    recovered_count = int(partial_dark_recovered or 0)
    if recovered_count > 0:
        for key in ("numero", "width", "height", "_band_id", "_band_y_top", "_band_index", "_source_page_number"):
            if (key not in ocr_page or ocr_page.get(key) in (None, "")) and key in page_dict:
                ocr_page[key] = page_dict[key]
        _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
        _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
        perf["ocr_partial_dark_bubble_recovered"] = recovered_count
        perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
    ocr_stats = ocr_page.get("_ocr_stats") if isinstance(ocr_page, dict) else None
    if isinstance(ocr_stats, dict) and bool(ocr_stats.get("scanlation_discord_promo_detected")):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        excluded_page = _mark_scanlation_discord_promo_band(ocr_page, perf)
        excluded_page["_vision_blocks"] = []
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=excluded_page,
        )
        _record_copyback_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=band.ocr_result,
            applied=False,
            reason="excluded_non_story_scanlation_discord_promo",
        )
        _finish(band.ocr_result)
        return band
    if not list(ocr_page.get("texts") or []):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result={**ocr_page, "texts": [], "_vision_blocks": []},
        )
        _record_copyback_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=band.ocr_result,
            applied=False,
            reason="no_texts",
        )
        _finish(band.ocr_result)
        return band
    if _ocr_page_is_scanlation_discord_promo_band(ocr_page, band):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        excluded_page = _mark_scanlation_discord_promo_band(ocr_page, perf)
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=excluded_page,
        )
        _record_copyback_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=band.ocr_result,
            applied=False,
            reason="excluded_non_story_scanlation_discord_promo",
        )
        _finish(band.ocr_result)
        return band
    if _smart_skip_real_enabled():
        _apply_smart_skip_real(ocr_page, perf)
    # Qualidade: Revisão contextual e enriquecimento de layout (SFX vs Fala, Balões Conectados)
    stage_start = time.perf_counter()
    review_layout_stage = _run_review_layout_stage(
        band,
        ocr_page=ocr_page,
        band_history=band_history,
        connected_reasoner_config=connected_reasoner_config,
        layout_page_image_bgr=layout_page_image_bgr,
        layout_page_y_top=layout_page_y_top,
    )
    ocr_page = review_layout_stage.to_page_dict()
    _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
    if list(ocr_page.get("texts") or []):
        post_layout_dark_recovered = _run_with_stage_lock(
            "ocr_post_layout_dark_lobe_recovery",
            ocr_lock,
            lambda: _recover_partial_dark_bubble_ocr_from_texts(
                ocr_page,
                band=band,
                page_dict=page_dict,
                band_id=band_id,
                idioma_origem=idioma_origem,
                runtime=runtime,
                work_title=obra,
                work_title_user_provided=work_title_user_provided,
                layout_lobes_only=True,
            ),
        )
        post_layout_recovered_count = int(post_layout_dark_recovered or 0)
        if post_layout_recovered_count > 0:
            perf["ocr_post_layout_dark_lobe_recovered"] = post_layout_recovered_count
            perf["ocr_text_count"] = int(len(ocr_page.get("texts") or []))
            _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
            _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
            review_layout_stage = _run_review_layout_stage(
                band,
                ocr_page=ocr_page,
                band_history=band_history,
                connected_reasoner_config=connected_reasoner_config,
                layout_page_image_bgr=layout_page_image_bgr,
                layout_page_y_top=layout_page_y_top,
            )
            ocr_page = review_layout_stage.to_page_dict()
            _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
    _normalize_dark_bubble_contracts_for_stage(ocr_page, band.strip_slice)
    dark_full_crop_replaced = _run_with_stage_lock(
        "ocr_dark_bubble_full_crop_reocr",
        ocr_lock,
        lambda: _replace_dark_bubble_text_with_full_crop_ocr(
            ocr_page,
            band,
            runtime=runtime,
            idioma_origem=idioma_origem,
            work_title=obra,
            work_title_user_provided=work_title_user_provided,
        ),
    )
    if int(dark_full_crop_replaced or 0) > 0:
        perf["ocr_dark_bubble_full_crop_replaced"] = int(dark_full_crop_replaced or 0)
        _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
        _record_ocr_raw_blocks(ocr_page, band=band, band_id=band_id)
        review_layout_stage = _run_review_layout_stage(
            band,
            ocr_page=ocr_page,
            band_history=band_history,
            connected_reasoner_config=connected_reasoner_config,
            layout_page_image_bgr=layout_page_image_bgr,
            layout_page_y_top=layout_page_y_top,
        )
        ocr_page = review_layout_stage.to_page_dict()
        _normalize_dark_bubble_contracts_for_stage(ocr_page, band.strip_slice)
        _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
    perf["source_style_evidence_applied"] = _apply_band_source_style_evidence(
        ocr_page,
        band.original_slice if band.original_slice is not None else band.strip_slice,
    )
    finalized_before_translate = _finalize_ocr_page_before_translation(
        ocr_page,
        band.strip_slice.shape if band.strip_slice.ndim == 3 else (band.strip_slice.shape[0], band.strip_slice.shape[1], 3),
        page_number=source_page_number,
        source_language=idioma_origem,
    )
    if finalized_before_translate:
        perf["ocr_finalized_before_translate"] = finalized_before_translate
        _attach_ocr_trace_metadata(ocr_page, band_id=band_id)
    _mark("review_layout", stage_start)

    if _smart_skip_shadow_enabled():
        _apply_smart_skip_shadow(ocr_page, perf)

    stage_start = time.perf_counter()
    translate_stage = _run_translate_stage(
        ocr_page,
        translator=translator,
        context=context,
        glossario=glossario,
        idioma_origem=idioma_origem,
        idioma_destino=idioma_destino,
        obra=obra,
        models_dir=models_dir,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        translation_context=translation_context,
    )
    _mark("translate", stage_start)

    translated_page = translate_stage.to_page_dict()
    _attach_ocr_trace_metadata(translated_page, band_id=band_id)
    if callable(ordered_context_after_translate_callback):
        ordered_context_after_translate_callback(copy.deepcopy(translated_page))

    inpaint_lock = inpaint_stage_lock if inpaint_stage_lock is not None else gpu_stage_lock
    inpaint_stage = _run_with_stage_lock(
        "inpaint",
        inpaint_lock,
        lambda: _run_inpaint_stage(
            band,
            inpainter=inpainter,
            translated_page=translated_page,
            band_index=page_idx + 1,
            source_page_number=source_page_number,
        ),
    )
    cleaned = inpaint_stage.to_image()
    perf.update(dict(inpaint_stage.perf_updates))
    typeset_stage = _run_with_stage_lock(
        "typeset",
        typeset_stage_lock,
        lambda: _run_typeset_stage(
            cleaned,
            typesetter=typesetter,
            translated_page=translated_page,
        ),
    )
    stage_start = time.perf_counter()
    copy_back_stage = _run_copy_back_stage(
        band,
        cleaned_slice=cleaned,
        rendered_slice=typeset_stage.to_image(),
        translated_page=translated_page,
    )
    _mark("copy_back", stage_start)
    _record_copyback_decision(
        band=band,
        band_id=band_id,
        source_page_number=source_page_number,
        translated_page=translated_page,
        applied=True,
        reason="copyback_outside_balloons",
    )
    _record_band_stage_visual_debug(
        band=band,
        band_id=band_id,
        source_page_number=source_page_number,
        post_typeset=typeset_stage.to_image(),
        post_copyback=copy_back_stage.to_image(),
    )
    _commit_band_outputs(
        band,
        cleaned_slice=cleaned,
        rendered_slice=copy_back_stage.to_image(),
        ocr_result=translated_page,
    )
    _finish(band.ocr_result)
    return band
