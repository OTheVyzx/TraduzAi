"""Reusa as stages existentes para processar uma Band como se fosse uma página."""

from __future__ import annotations

import copy
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
    for index, text in enumerate(texts):
        text_id = _text_id_for(text, index)
        text["id"] = str(text.get("id") or text_id)
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
        for txt in ocr_page.get("texts", []):
            bbox = txt.get("balloon_bbox") or txt.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            _mark(
                bx1 - balloon_margin,
                by1 - balloon_margin,
                bx2 + balloon_margin,
                by2 + balloon_margin,
            )
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
    ):
        if (key not in merged_page or merged_page.get(key) in (None, "")) and key in ocr_page:
            merged_page[key] = ocr_page[key]

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
        bright_ratio = float(evidence.get("bright_pixel_ratio") or 0.0)
        dark_ratio = float(evidence.get("dark_pixel_ratio") or 0.0)
    except Exception:
        return False
    return (
        bool(evidence.get("has_inner_dark_text"))
        and significant_count >= 2
        and significant_area >= 300
        and bright_ratio >= 0.25
        and dark_ratio <= 0.12
    )


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
    if len(compact) <= 3 and any(ch.isdigit() for ch in compact):
        return False
    if len(compact) <= 3 and len(set(compact.lower())) <= 1:
        return False
    return True


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
        return False
    if bbox_w > int(crop_width * 0.86) and bbox_h > int(crop_height * 0.28):
        return False
    return True


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
        if _inner_dark_text_evidence is not None:
            evidence = _inner_dark_text_evidence(image, BBox(x1, y1, x2, y2))
            if not _candidate_crop_reocr_evidence_is_strong(evidence):
                continue
        pad_x = 20
        pad_y = max(40, int(round((y2 - y1) * 0.7)))
        crop_left = max(0, x1 - pad_x)
        crop_top = max(0, y1 - pad_y)
        crop_right = min(width, x2 + pad_x)
        crop_bottom = min(height, y2 + pad_y)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            continue
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
        texts, blocks = _map_crop_ocr_page_to_band(
            crop_result,
            band_page=page_dict,
            band_id=band_id,
            balloon_local_bbox=[x1, y1, x2, y2],
            crop_left=crop_left,
            crop_top=crop_top,
            candidate_index=candidate_index,
        )
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
    return BandStageOutput(
        "ocr_candidate_recovery",
        page,
        {
            "ocr_candidate_crop_candidates": int(candidate_count),
            "ocr_candidate_crop_attempts": int(attempts),
            "ocr_candidate_crop_recovered": int(len(recovered_texts)),
        },
    )


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


def _ensure_text_balloon_bboxes(page: dict, band: Band) -> None:
    vision_blocks = page.get("_vision_blocks", [])
    for txt in page.get("texts", []):
        tx1, ty1, tx2, ty2 = txt.get("bbox", [0, 0, 0, 0])
        best = None
        best_iou = 0.0
        for vb in vision_blocks:
            vx1, vy1, vx2, vy2 = vb["bbox"]
            ix = max(0, min(tx2, vx2) - max(tx1, vx1))
            iy = max(0, min(ty2, vy2) - max(ty1, vy1))
            inter = ix * iy
            ta = max(1, (tx2 - tx1) * (ty2 - ty1))
            ratio = inter / ta
            if ratio > best_iou:
                best_iou = ratio
                best = vb
        if best:
            if not txt.get("balloon_bbox"):
                txt["balloon_bbox"] = list(best["bbox"])
            for key in ("bubble_id", "bubble_mask_bbox", "bubble_inner_bbox"):
                if best.get(key) not in (None, [], "") and txt.get(key) in (None, [], ""):
                    txt[key] = copy.deepcopy(best[key])
        else:
            if txt.get("balloon_bbox"):
                continue
            w = page.get("width", band.strip_slice.shape[1])
            h = page.get("height", band.strip_slice.shape[0])
            txt["balloon_bbox"] = [
                max(0, tx1 - 8), max(0, ty1 - 8),
                min(w, tx2 + 8), min(h, ty2 + 8),
            ]


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
    page_for_typeset = _without_legacy_decision_fields_for_stage(translated_page)
    return BandImageStageOutput(
        "typeset",
        typesetter.render_band_image(cleaned_slice, page_for_typeset),
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

    if _all_translations_unchanged(translated_page):
        original = band.original_slice if band.original_slice is not None else band.strip_slice
        _ensure_text_balloon_bboxes(translated_page, band)
        _commit_band_outputs(
            band,
            cleaned_slice=original,
            rendered_slice=original,
            ocr_result=translated_page,
        )
        perf["unchanged_translation_skip"] = True
        _record_copyback_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=translated_page,
            applied=False,
            reason="unchanged_translation_skip",
        )
        _record_inpaint_skip_decision(
            band=band,
            band_id=band_id,
            source_page_number=source_page_number,
            translated_page=translated_page,
            reason="unchanged_translation_skip",
        )
        _finish(band.ocr_result)
        return band

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
    _commit_band_outputs(
        band,
        cleaned_slice=cleaned,
        rendered_slice=copy_back_stage.to_image(),
        ocr_result=translated_page,
    )
    _finish(band.ocr_result)
    return band
