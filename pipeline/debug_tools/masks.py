from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

MASK_DENSITY_WARN = 0.12
MASK_DENSITY_BORDERLINE_WARN = 0.15
MASK_DENSITY_STRONG_WARN = 0.30
EXPANDED_RAW_WARN = 2.5
SOURCE_GLYPH_REVIEW = 1.5
SOURCE_GLYPH_CRITICAL = 8.0
OUTSIDE_BALLOON_WARN_RATIO = 0.08
CLEAN_OUTSIDE_BALLOON_RATIO = 0.01
CLEAN_EXPANDED_RAW_RATIO = 1.5
CLEAN_SOURCE_GLYPH_RATIO = 1.2
OUTSIDE_BALLOON_CRITICAL_PIXELS = 50
OUTSIDE_BALLOON_CRITICAL_RATIO = 0.18

_MASK_SUMMARY_STATE: dict[int, dict[str, dict[str, Any]]] = {}


def _image_hw(image: np.ndarray) -> tuple[int, int]:
    return int(image.shape[0]), int(image.shape[1])


def _blank_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def _as_mask(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return _blank_mask(height, width)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _normalize_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _normalize_polygon(value: Any, width: int, height: int) -> list[list[int]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
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


def _normalize_polygons(value: Any, width: int, height: int) -> list[list[list[int]]]:
    if not isinstance(value, (list, tuple)) or not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and not (
        first and isinstance(first[0], (list, tuple))
    ):
        polygon = _normalize_polygon(value, width, height)
        return [polygon] if polygon else []
    polygons: list[list[list[int]]] = []
    for item in value:
        polygon = _normalize_polygon(item, width, height)
        if polygon:
            polygons.append(polygon)
    return polygons


def _fill_polygons(mask: np.ndarray, polygons: list[list[list[int]]]) -> None:
    for polygon in polygons:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)


def _mask_from_bbox(width: int, height: int, bbox: list[int] | None) -> np.ndarray:
    mask = _blank_mask(height, width)
    if bbox:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    return mask


def _mask_from_polygons(width: int, height: int, polygons: list[list[list[int]]]) -> np.ndarray:
    mask = _blank_mask(height, width)
    _fill_polygons(mask, polygons)
    return mask


def _union_mask_from_texts(texts: list[dict[str, Any]], width: int, height: int, kind: str) -> np.ndarray:
    mask = _blank_mask(height, width)
    for text in texts:
        if kind in {"glyph", "line"}:
            _fill_polygons(mask, _normalize_polygons(text.get("line_polygons"), width, height))
        elif kind == "detected":
            bbox = _normalize_bbox(text.get("text_pixel_bbox"), width, height) or _normalize_bbox(text.get("bbox"), width, height)
            mask |= _mask_from_bbox(width, height, bbox)
        elif kind == "balloon":
            polygons = _normalize_polygons(text.get("balloon_polygon"), width, height)
            if polygons:
                _fill_polygons(mask, polygons)
            else:
                mask |= _mask_from_bbox(width, height, _normalize_bbox(text.get("balloon_bbox"), width, height))
    return mask


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area(bbox: list[int] | None) -> int:
    if not bbox:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _source_bbox_area(texts: list[dict[str, Any]], width: int, height: int) -> int:
    mask = _blank_mask(height, width)
    for text in texts:
        mask |= _mask_from_bbox(width, height, _normalize_bbox(text.get("bbox"), width, height))
    return int(np.count_nonzero(mask))


def _bboxes_nearly_equal(first: list[int] | None, second: list[int] | None, tolerance: int = 2) -> bool:
    if first is None or second is None:
        return False
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(first, second))


def _has_explicit_balloon_shape(text: dict[str, Any], width: int, height: int) -> bool:
    if _normalize_polygons(text.get("balloon_polygon"), width, height):
        return True
    for key in (
        "balloon_subregions",
        "connected_lobe_bboxes",
        "connected_lobe_polygons",
        "connected_position_bboxes",
    ):
        raw = text.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            return True
    return False


def _reference_bboxes_for_text(text: dict[str, Any], width: int, height: int) -> list[list[int]]:
    refs: list[list[int]] = []
    for key in ("bbox", "source_bbox", "text_pixel_bbox", "ocr_text_bbox"):
        bbox = _normalize_bbox(text.get(key), width, height)
        if bbox and bbox not in refs:
            refs.append(bbox)
    return refs


def _uses_synthetic_tight_balloon_reference(
    texts: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    mask_source: str,
    used_balloon_clip: bool,
    source_glyph_area_ratio: float,
    mask_balloon_ratio: float,
) -> bool:
    if not used_balloon_clip or mask_source != "line_polygons" or not texts:
        return False
    if any(_has_explicit_balloon_shape(text, width, height) for text in texts):
        return False

    checked = 0
    for text in texts:
        balloon = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if not balloon:
            continue
        refs = _reference_bboxes_for_text(text, width, height)
        if not refs:
            return False
        checked += 1
        if any(_bboxes_nearly_equal(balloon, ref) for ref in refs):
            continue

        balloon_area = _bbox_area(balloon)
        ref_area = max(_bbox_area(ref) for ref in refs)
        if (
            ref_area > 0
            and balloon_area <= int(ref_area * 1.35)
            and source_glyph_area_ratio <= 1.40
            and mask_balloon_ratio <= 1.65
        ):
            continue
        return False
    return checked > 0


def _uses_edge_clipped_text_bbox_reference(
    texts: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    mask_source: str,
    used_balloon_clip: bool,
    outside_balloon_ratio: float,
    mask_balloon_ratio: float,
    expanded_raw_ratio: float,
    has_bbox_overreach: bool,
) -> bool:
    if not used_balloon_clip or mask_source not in {"text_pixel_bbox", "bbox"} or not texts:
        return False
    if has_bbox_overreach:
        return False
    if outside_balloon_ratio > 0.24 or mask_balloon_ratio > 1.35 or expanded_raw_ratio > 1.35:
        return False
    if any(_has_explicit_balloon_shape(text, width, height) for text in texts):
        return False

    checked = 0
    for text in texts:
        balloon = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if not balloon:
            return False
        refs = _reference_bboxes_for_text(text, width, height)
        if not refs:
            return False
        checked += 1
        if any(_bboxes_nearly_equal(balloon, ref) for ref in refs):
            continue

        balloon_area = _bbox_area(balloon)
        ref_area = max(_bbox_area(ref) for ref in refs)
        if balloon_area <= 0 or ref_area <= 0:
            return False
        area_ratio = max(balloon_area, ref_area) / float(max(1, min(balloon_area, ref_area)))
        if area_ratio <= 1.35:
            continue
        return False
    return checked > 0


def _text_ids(texts: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts, start=1):
        raw_id = text.get("id") or text.get("text_id") or text.get("_id")
        ids.append(str(raw_id or f"text_{index:03d}"))
    return ids


def _trace_ids(texts: list[dict[str, Any]], text_ids: list[str], band_id: str) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts):
        text_id = text_ids[index] if index < len(text_ids) else str(text.get("id") or text.get("text_id") or "")
        raw_id = text.get("trace_id") or (f"{text_id}@{band_id}" if text_id and band_id else None)
        trace_id = str(raw_id or "").strip()
        if trace_id and trace_id not in ids:
            ids.append(trace_id)
    return ids


def _text_instance_ids(texts: list[dict[str, Any]], text_ids: list[str], band_id: str) -> list[str]:
    ids: list[str] = []
    for index, text in enumerate(texts):
        text_id = text_ids[index] if index < len(text_ids) else str(text.get("id") or text.get("text_id") or "")
        raw_id = (
            text.get("text_instance_id")
            or text.get("instance_id")
            or (f"{band_id}_{text_id}" if text_id and band_id else None)
        )
        instance_id = str(raw_id or "").strip()
        if instance_id and instance_id not in ids:
            ids.append(instance_id)
    return ids


def _band_id(ocr_page: dict[str, Any]) -> str:
    for text in [item for item in ocr_page.get("texts", []) if isinstance(item, dict)]:
        for key in ("band_id", "_band_id"):
            raw_band_id = str(text.get(key) or "").strip()
            if raw_band_id:
                return raw_band_id
        trace_id = str(text.get("trace_id") or "").strip()
        if "@" in trace_id:
            trace_band_id = trace_id.rsplit("@", 1)[-1].strip()
            if trace_band_id:
                return trace_band_id
    for key in ("_band_id", "band_id"):
        raw_band_id = str(ocr_page.get(key) or "").strip()
        if raw_band_id:
            return raw_band_id
    try:
        page_number = int(ocr_page.get("_source_page_number") or ocr_page.get("numero") or 0)
    except Exception:
        page_number = 0
    try:
        band_index = int(ocr_page.get("_band_index") or 0)
    except Exception:
        band_index = 0
    return f"page_{page_number:03d}_band_{band_index:03d}"


def _mask_source(texts: list[dict[str, Any]]) -> str:
    if any(_normalize_polygons(text.get("line_polygons"), 1_000_000, 1_000_000) for text in texts):
        return "line_polygons"
    if any(text.get("text_pixel_bbox") for text in texts):
        return "text_pixel_bbox"
    if any(text.get("bbox") for text in texts):
        return "bbox"
    return "fallback"


def _text_profile_values(texts: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for text in texts:
        for key in ("content_class", "tipo", "layout_profile", "block_profile", "balloon_type"):
            raw = str(text.get(key) or "").strip().lower()
            if raw:
                values.add(raw)
    return values


def _clean_line_polygon_mask(
    *,
    mask_source: str,
    outside_balloon_ratio: float,
    expanded_raw_ratio: float,
) -> bool:
    return (
        mask_source == "line_polygons"
        and outside_balloon_ratio <= CLEAN_OUTSIDE_BALLOON_RATIO
        and expanded_raw_ratio <= CLEAN_EXPANDED_RAW_RATIO
    )


def _mask_density_high_gate(
    texts: list[dict[str, Any]],
    *,
    mask_source: str,
    mask_density: float,
    outside_balloon_ratio: float,
    expanded_raw_ratio: float,
    source_glyph_area_ratio: float,
    has_bbox_overreach: bool,
    synthetic_tight_balloon_reference: bool = False,
) -> bool:
    if mask_density <= MASK_DENSITY_WARN:
        return False
    if (
        synthetic_tight_balloon_reference
        and mask_source == "line_polygons"
        and source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO
        and expanded_raw_ratio <= CLEAN_EXPANDED_RAW_RATIO
    ):
        return False
    if outside_balloon_ratio >= OUTSIDE_BALLOON_WARN_RATIO:
        return True
    if expanded_raw_ratio > EXPANDED_RAW_WARN or source_glyph_area_ratio >= SOURCE_GLYPH_CRITICAL:
        return True
    if has_bbox_overreach:
        return True
    if mask_source != "line_polygons":
        return True

    profiles = _text_profile_values(texts)
    clean_line_mask = _clean_line_polygon_mask(
        mask_source=mask_source,
        outside_balloon_ratio=outside_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
    )
    if clean_line_mask and profiles & {"narracao", "narration", "top_narration", "dark", "textured"}:
        if source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO:
            return False
    if clean_line_mask and mask_density < MASK_DENSITY_BORDERLINE_WARN and source_glyph_area_ratio <= CLEAN_SOURCE_GLYPH_RATIO:
        return False
    if mask_density >= MASK_DENSITY_STRONG_WARN and source_glyph_area_ratio >= SOURCE_GLYPH_REVIEW:
        return True
    return False


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.uint8).copy()
    if overlay.ndim != 3 or overlay.shape[2] < 3:
        overlay = np.repeat(_as_mask(mask, *_image_hw(image_rgb))[:, :, None], 3, axis=2)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    active = mask > 0
    overlay[active] = (overlay[active].astype(np.float32) * 0.6 + red[active].astype(np.float32) * 0.4).astype(np.uint8)
    return overlay


def build_mask_chain_debug_payload(
    ocr_page: dict[str, Any],
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    final_mask: np.ndarray | None = None,
    protection_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    height, width = _image_hw(image_rgb)
    texts = [text for text in ocr_page.get("texts", []) if isinstance(text, dict)]
    glyph_mask = _union_mask_from_texts(texts, width, height, "glyph")
    line_polygon_mask = _union_mask_from_texts(texts, width, height, "line")
    detected_text_mask = _union_mask_from_texts(texts, width, height, "detected")
    balloon_mask = _union_mask_from_texts(texts, width, height, "balloon")
    if np.any(balloon_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        balloon_inner_mask = cv2.erode(balloon_mask, kernel, iterations=1)
    else:
        balloon_inner_mask = _blank_mask(height, width)
    raw = _as_mask(raw_mask, height, width)
    expanded = _as_mask(expanded_mask, height, width)
    protection = _as_mask(protection_mask, height, width)
    final = _as_mask(final_mask, height, width) if final_mask is not None else expanded.copy()
    if np.any(protection):
        final = np.where((final > 0) & (protection == 0), 255, 0).astype(np.uint8)

    outside_balloon_pixels = 0
    outside_reference = final if final_mask is not None or np.any(protection) else expanded
    used_balloon_clip = bool(np.any(balloon_mask))
    if used_balloon_clip:
        outside_balloon_pixels = int(np.count_nonzero((outside_reference > 0) & (balloon_mask == 0)))

    raw_pixels = int(np.count_nonzero(raw))
    expanded_pixels = int(np.count_nonzero(expanded))
    final_pixels = int(np.count_nonzero(final))
    balloon_pixels = int(np.count_nonzero(balloon_mask))
    outside_reference_pixels = int(np.count_nonzero(outside_reference))
    outside_balloon_ratio = round(outside_balloon_pixels / float(max(1, outside_reference_pixels)), 6)
    band_pixels = max(1, int(height * width))
    mask_density = round(expanded_pixels / float(band_pixels), 6)
    mask_balloon_ratio = round(final_pixels / float(max(1, balloon_pixels)), 6)
    expanded_raw_ratio = round(expanded_pixels / float(max(1, raw_pixels)), 6)
    source_area = _source_bbox_area(texts, width, height)
    glyph_bbox_area = _bbox_area(_bbox_from_mask(glyph_mask))
    source_glyph_area_ratio = round(source_area / float(max(1, glyph_bbox_area)), 6)
    mask_source = _mask_source(texts)
    has_bbox_overreach = any("bbox_overreach" in (text.get("qa_flags") or []) for text in texts)
    has_bbox_overreach_critical = any("bbox_overreach_critical" in (text.get("qa_flags") or []) for text in texts)
    synthetic_tight_balloon_reference = _uses_synthetic_tight_balloon_reference(
        texts,
        width,
        height,
        mask_source=mask_source,
        used_balloon_clip=used_balloon_clip,
        source_glyph_area_ratio=source_glyph_area_ratio,
        mask_balloon_ratio=mask_balloon_ratio,
    )
    edge_clipped_text_bbox_reference = _uses_edge_clipped_text_bbox_reference(
        texts,
        width,
        height,
        mask_source=mask_source,
        used_balloon_clip=used_balloon_clip,
        outside_balloon_ratio=outside_balloon_ratio,
        mask_balloon_ratio=mask_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
        has_bbox_overreach=has_bbox_overreach or has_bbox_overreach_critical,
    )
    mask_density_high = _mask_density_high_gate(
        texts,
        mask_source=mask_source,
        mask_density=mask_density,
        outside_balloon_ratio=outside_balloon_ratio,
        expanded_raw_ratio=expanded_raw_ratio,
        source_glyph_area_ratio=source_glyph_area_ratio,
        has_bbox_overreach=has_bbox_overreach or has_bbox_overreach_critical,
        synthetic_tight_balloon_reference=synthetic_tight_balloon_reference,
    )
    outside_balloon_critical = (
        outside_balloon_pixels > OUTSIDE_BALLOON_CRITICAL_PIXELS
        and outside_balloon_ratio >= OUTSIDE_BALLOON_CRITICAL_RATIO
        and not synthetic_tight_balloon_reference
        and not edge_clipped_text_bbox_reference
    )
    gates = {
        "mask_density_high": mask_density_high,
        "mask_outside_balloon": (
            outside_balloon_pixels > 0
            and outside_balloon_ratio >= OUTSIDE_BALLOON_WARN_RATIO
        ),
        "mask_outside_balloon_critical": outside_balloon_critical,
        "bbox_overreach": has_bbox_overreach,
        "bbox_overreach_critical": has_bbox_overreach_critical,
        "expanded_ratio_review": expanded_raw_ratio > EXPANDED_RAW_WARN,
    }
    flags = [name for name, enabled in gates.items() if enabled]
    ids = _text_ids(texts)
    band_id = _band_id(ocr_page)
    trace_ids = _trace_ids(texts, ids, band_id)
    text_instance_ids = _text_instance_ids(texts, ids, band_id)
    decision = {
        "schema_version": 1,
        "band_id": band_id,
        "text_id": ids[0] if len(ids) == 1 else None,
        "text_ids": ids,
        "trace_ids": trace_ids,
        "trace_ids_in_band": trace_ids,
        "text_instance_ids": text_instance_ids,
        "mask_source": mask_source,
        "used_balloon_clip": used_balloon_clip,
        "synthetic_tight_balloon_reference": synthetic_tight_balloon_reference,
        "edge_clipped_text_bbox_reference": edge_clipped_text_bbox_reference,
        "used_protection_mask": bool(np.any(protection)),
        "raw_mask_pixels": raw_pixels,
        "expanded_mask_pixels": expanded_pixels,
        "final_mask_pixels": final_pixels,
        "balloon_mask_pixels": balloon_pixels,
        "balloon_inner_mask_pixels": int(np.count_nonzero(balloon_inner_mask)),
        "mask_balloon_ratio": mask_balloon_ratio,
        "outside_balloon_pixels": outside_balloon_pixels,
        "outside_balloon_ratio": outside_balloon_ratio,
        "outside_balloon_reference": "final_mask" if outside_reference is final else "expanded_mask",
        "expanded_raw_ratio": expanded_raw_ratio,
        "mask_density_in_band": mask_density,
        "source_bbox_area": source_area,
        "glyph_bbox_area": glyph_bbox_area,
        "source_glyph_area_ratio": source_glyph_area_ratio,
        "flags": flags,
        "gates": gates,
        "thresholds": {
            "mask_density_warn": MASK_DENSITY_WARN,
            "mask_density_borderline_warn": MASK_DENSITY_BORDERLINE_WARN,
            "mask_density_strong_warn": MASK_DENSITY_STRONG_WARN,
            "expanded_raw_warn": EXPANDED_RAW_WARN,
            "source_glyph_review": SOURCE_GLYPH_REVIEW,
            "source_glyph_critical": SOURCE_GLYPH_CRITICAL,
            "outside_balloon_warn_ratio": OUTSIDE_BALLOON_WARN_RATIO,
            "clean_outside_balloon_ratio": CLEAN_OUTSIDE_BALLOON_RATIO,
            "clean_expanded_raw_ratio": CLEAN_EXPANDED_RAW_RATIO,
            "clean_source_glyph_ratio": CLEAN_SOURCE_GLYPH_RATIO,
            "outside_balloon_critical_pixels": OUTSIDE_BALLOON_CRITICAL_PIXELS,
            "outside_balloon_critical_ratio": OUTSIDE_BALLOON_CRITICAL_RATIO,
        },
    }
    images = {
        "01_glyph_mask.png": glyph_mask,
        "02_line_polygon_mask.png": line_polygon_mask,
        "03_detected_text_mask.png": detected_text_mask,
        "04_balloon_mask.png": balloon_mask,
        "05_balloon_inner_mask.png": balloon_inner_mask,
        "06_protection_mask.png": protection,
        "07_raw_text_mask.png": raw,
        "08_expanded_text_mask.png": expanded,
        "09_final_inpaint_mask.png": final,
        "10_mask_overlay.jpg": _overlay_mask(image_rgb, final),
    }
    return decision, images


def _summary_from_decisions(decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    flagged_bands: list[str] = []
    totals = {
        "raw_mask_pixels": 0,
        "expanded_mask_pixels": 0,
        "outside_balloon_pixels": 0,
    }
    bands_with_mask = 0
    for band_id, decision in sorted(decisions.items()):
        if int(decision.get("expanded_mask_pixels") or 0) > 0:
            bands_with_mask += 1
        if decision.get("flags"):
            flagged_bands.append(band_id)
        by_source[str(decision.get("mask_source") or "unknown")] += 1
        for key in totals:
            totals[key] += int(decision.get(key) or 0)
    return {
        "schema_version": 1,
        "band_count": len(decisions),
        "bands_with_mask": bands_with_mask,
        "bands_with_flags": len(flagged_bands),
        "totals": totals,
        "by_source": dict(sorted(by_source.items())),
        "flagged_bands": flagged_bands,
    }


def _recorder_key(recorder: Any) -> int:
    return id(recorder)


def _write_recorder_image(recorder: Any, rel_path: str, image: np.ndarray) -> None:
    output = image
    if output.ndim == 3 and output.shape[2] >= 3:
        output = cv2.cvtColor(output[:, :, :3], cv2.COLOR_RGB2BGR)
    recorder.write_image(rel_path, output)


def write_mask_chain_debug_artifacts(
    recorder: Any,
    ocr_page: dict[str, Any],
    *,
    image_rgb: np.ndarray,
    raw_mask: np.ndarray | None,
    expanded_mask: np.ndarray | None,
    final_mask: np.ndarray | None = None,
    protection_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    try:
        decision, images = build_mask_chain_debug_payload(
            ocr_page,
            image_rgb=image_rgb,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
            final_mask=final_mask,
            protection_mask=protection_mask,
        )
        band_id = str(decision["band_id"])
        base = f"06_mask_segmentation/{band_id}"
        for filename, image in images.items():
            _write_recorder_image(recorder, f"{base}/{filename}", image)
        recorder.write_json(f"{base}/mask_decision.json", decision)
        state = _MASK_SUMMARY_STATE.setdefault(_recorder_key(recorder), {})
        state[band_id] = decision
        recorder.write_json("06_mask_segmentation/mask_chain_summary.json", _summary_from_decisions(state))
        return decision
    except Exception as exc:
        try:
            recorder.event(
                "mask_segmentation",
                "mask_chain_debug_failed",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        return None
