"""Adapter em-memoria do inpainter para o pipeline strip-based."""

from __future__ import annotations

import os
import time

import numpy as np
import cv2


def _normalize_bbox(raw_bbox, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _build_fallback_vision_blocks(ocr_page: dict, width: int, height: int) -> list[dict]:
    blocks: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for txt in ocr_page.get("texts", []):
        bbox = (
            _normalize_bbox(txt.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(txt.get("bbox"), width, height)
            or _normalize_bbox(txt.get("balloon_bbox"), width, height)
        )
        if bbox is None:
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            {
                "bbox": bbox,
                "mask": None,
                "confidence": float(txt.get("confidence", txt.get("ocr_confidence", 0.0)) or 0.0),
                "text_pixel_bbox": txt.get("text_pixel_bbox"),
                "line_polygons": txt.get("line_polygons"),
                "balloon_type": txt.get("balloon_type"),
                "block_profile": txt.get("block_profile"),
            }
        )
    return blocks


def _fast_white_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_INPAINT", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_white_post_cleanup_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_POST_CLEANUP", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _fast_white_narration_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_WHITE_NARRATION", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_local_balloon_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_LOCAL_INPAINT", "0").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _fast_metadata_background_fill_enabled() -> bool:
    flag = os.getenv("TRADUZAI_STRIP_FAST_METADATA_FILL", "1").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _text_allows_fast_white_fill(text: dict) -> bool:
    return not _fast_white_rejection_reason(text)


def _fast_white_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict) or text.get("skip_processing"):
        return "skip_processing"
    tipo = str(text.get("tipo") or "fala").strip().lower()
    balloon_type = str(text.get("balloon_type") or "").strip().lower()
    profiles = {
        str(text.get("layout_profile") or "").strip().lower(),
        str(text.get("block_profile") or "").strip().lower(),
    }
    if profiles & {"connected_balloon"}:
        return "connected_balloon"
    has_contextual_neighbor = bool(str(text.get("context_before") or "").strip()) or bool(
        str(text.get("context_after") or "").strip()
    )
    if has_contextual_neighbor and bool(profiles & {"white_balloon"}):
        return "contextual_white_balloon"

    raw_confidence = text.get("ocr_confidence", text.get("confidence"))
    if raw_confidence is not None:
        try:
            confidence = float(raw_confidence)
        except Exception:
            confidence = 1.0
        if confidence < 0.85:
            moderate_clean_white = (
                confidence >= 0.75
                and balloon_type == "white"
                and bool(profiles & {"white_balloon", "top_narration"})
                and bool(text.get("text_pixel_bbox") or text.get("line_polygons"))
            )
            if not moderate_clean_white:
                return "low_confidence"

    if tipo in {"fala", "pensamento"}:
        return ""
    if tipo != "narracao":
        return "unsupported_tipo"
    if not _fast_white_narration_enabled():
        return "narration_disabled"

    if balloon_type == "white" and bool(profiles & {"white_balloon", "top_narration"}):
        return ""
    return "unsupported_narration"


def _edge_clipped_white_balloon_fallback_bbox(
    image_rgb: np.ndarray,
    text: dict,
    balloon_bbox: list[int],
) -> list[int] | None:
    from vision_stack.runtime import _extract_white_balloon_fill_mask

    if not isinstance(text, dict):
        return None
    tipo = str(text.get("tipo") or "fala").strip().lower()
    if tipo not in {"fala", "pensamento", "narracao"}:
        return None

    height, width = image_rgb.shape[:2]
    bbox = _normalize_bbox(balloon_bbox, width, height)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    edge_margin = max(16, int(round(min(height, width) * 0.04)))
    touches_crop_edge = y1 <= edge_margin or y2 >= (height - edge_margin)
    if not touches_crop_edge:
        return None

    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    bright_ratio = float(np.mean(gray >= 220))
    p80 = float(np.percentile(gray, 80))
    if bright_ratio < 0.72 or p80 < 228.0:
        return None

    bbox_area = max(1, box_w * box_h)
    pad = max(8, min(edge_margin, 32))
    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)
    if rx2 > rx1 and ry2 > ry1:
        context = image_rgb[ry1:ry2, rx1:rx2]
        ring = np.ones((ry2 - ry1, rx2 - rx1), dtype=bool)
        ring[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
        if int(np.count_nonzero(ring)) >= 64:
            ring_gray = cv2.cvtColor(context, cv2.COLOR_RGB2GRAY)[ring]
            ring_bright_ratio = float(np.mean(ring_gray >= 210))
            bbox_area_ratio = bbox_area / float(max(1, width * height))
            if bbox_area_ratio >= 0.20 and ring_bright_ratio < 0.35:
                return None

    fill_mask = _extract_white_balloon_fill_mask(image_rgb, bbox)
    if not isinstance(fill_mask, np.ndarray) or fill_mask.size == 0:
        return None
    fill_region = fill_mask[y1:y2, x1:x2]
    fill_ratio = float(np.mean(fill_region > 0))
    if fill_ratio < 0.94:
        return None

    return bbox


def _bbox_center_inside(inner: list[int], outer: list[int]) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(min(a_area, b_area))


def _block_is_covered_by_fast_fill(block: dict, filled_bboxes: list[list[int]], width: int, height: int) -> bool:
    bbox = _normalize_bbox(block.get("bbox"), width, height)
    if bbox is None:
        return False
    for filled in filled_bboxes:
        if _bbox_center_inside(bbox, filled) or _bbox_overlap_ratio(bbox, filled) >= 0.25:
            return True
    return False


def _apply_fast_white_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_white_balloon_count"] = stats["white_balloon_count"]
        ocr_page["_strip_remaining_inpaint_blocks"] = stats["remaining_blocks"]
        ocr_page["_strip_fast_white_rejection_reasons"] = dict(rejection_reasons)
        return stats

    if not _fast_white_balloon_fill_enabled():
        text_count = len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)])
        rejection_reasons["disabled"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"white_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    from vision_stack.runtime import _apply_white_balloon_fill, _resolve_white_balloon_bbox

    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []

    for text in ocr_page.get("texts", []):
        rejection_reason = _fast_white_rejection_reason(text)
        if rejection_reason:
            _reject(rejection_reason)
            continue
        balloon_bbox = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if balloon_bbox is None:
            _reject("missing_balloon_bbox")
            continue
        text_bbox = (
            _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue

        # Require balloon_bbox so the fast path only runs after layout enrichment,
        # but seed the mask from the text pixels. Strip bands often clip the
        # balloon bbox at the band edge, making it much broader than the text.
        resolved = _resolve_white_balloon_bbox(
            band_rgb,
            {
                "bbox": text_bbox,
                "text_pixel_bbox": text_bbox,
            },
        )
        if resolved is None:
            resolved = _edge_clipped_white_balloon_fallback_bbox(band_rgb, text, balloon_bbox)
        if resolved is None:
            _reject("no_white_fill_mask")
            continue
        filled_from_original = _apply_white_balloon_fill(band_rgb, resolved)
        changed_mask = np.any(filled_from_original != band_rgb, axis=2)
        result[changed_mask] = filled_from_original[changed_mask]
        filled_bboxes.append(resolved)

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record({"white_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height)
    ]
    stats = {
        "white_balloon_count": len(filled_bboxes),
        "remaining_blocks": len(remaining_blocks),
    }
    return result, remaining_blocks, _record(stats)


def _text_allows_fast_local_fill(text: dict) -> bool:
    return not _fast_local_rejection_reason(text)


def _fast_local_rejection_reason(text: dict) -> str:
    if not isinstance(text, dict) or text.get("skip_processing"):
        return "skip_processing"
    tipo = str(text.get("tipo") or "fala").strip().lower()
    if tipo in {"fala", "pensamento", "narracao"}:
        return ""
    return "unsupported_tipo"


def _mask_from_bbox(width: int, height: int, bbox: list[int], padding: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _expanded_bbox(width: int, height: int, bbox: list[int], padding: int) -> list[int] | None:
    x1, y1, x2, y2 = bbox
    return _normalize_bbox([x1 - padding, y1 - padding, x2 + padding, y2 + padding], width, height)


def _try_solid_background_text_fill(
    image_rgb: np.ndarray,
    text_bbox: list[int],
    fill_bbox: list[int],
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    text_bbox = _normalize_bbox(text_bbox, width, height)
    fill_bbox = _normalize_bbox(fill_bbox, width, height)
    if text_bbox is None or fill_bbox is None:
        return None

    context_bbox = [
        min(fill_bbox[0], text_bbox[0] - 24),
        min(fill_bbox[1], text_bbox[1] - 24),
        max(fill_bbox[2], text_bbox[2] + 24),
        max(fill_bbox[3], text_bbox[3] + 24),
    ]
    context_bbox = _normalize_bbox(context_bbox, width, height)
    if context_bbox is None:
        return None

    cx1, cy1, cx2, cy2 = context_bbox
    local = image_rgb[cy1:cy2, cx1:cx2]
    if local.size == 0:
        return None

    tx1, ty1, tx2, ty2 = text_bbox
    local_text_bbox = [tx1 - cx1, ty1 - cy1, tx2 - cx1, ty2 - cy1]
    text_mask = _mask_from_bbox(local.shape[1], local.shape[0], local_text_bbox, padding=8)
    sample = local[text_mask == 0]
    if sample.size == 0 or len(sample) < 64:
        return None

    sample_f = sample.astype(np.float32)
    median = np.median(sample_f, axis=0)
    std = np.sqrt(np.mean(np.square(sample_f - median[None, :]), axis=0))
    median_luma = float(np.mean(median))
    text_area = max(1, (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1]))
    max_std = max(float(v) for v in std)
    dark_panel_sample = False
    if text_area > 24_000 and median_luma <= 12.0:
        sample_luma = np.mean(sample_f, axis=1)
        dark_panel_sample = (
            max_std <= 16.0
            and float(np.percentile(sample_luma, 90)) <= 28.0
            and float(np.percentile(sample_luma, 98)) <= 80.0
        )
    if max_std > 10.0 and not dark_panel_sample:
        return None
    if median_luma <= 32.0:
        if text_area > 24_000:
            return None
    elif median_luma >= 238.0:
        pass
    else:
        return None

    region = image_rgb[ty1:ty2, tx1:tx2].astype(np.float32)
    if region.size == 0:
        return None
    contrast = float(np.max(np.abs(region - median[None, None, :])))
    if contrast < 32.0:
        return None

    bbox_width = text_bbox[2] - text_bbox[0]
    bbox_height = text_bbox[3] - text_bbox[1]
    fill_padding = max(8, min(24, int(round(max(bbox_width, bbox_height) * 0.08))))
    fill_bbox = _expanded_bbox(width, height, text_bbox, padding=fill_padding)
    if fill_bbox is None:
        return None
    fx1, fy1, fx2, fy2 = fill_bbox
    result = image_rgb.copy()
    fill = np.asarray([int(round(float(v))) for v in median], dtype=np.uint8)
    result[fy1:fy2, fx1:fx2] = fill
    return result


def _metadata_background_color(text: dict) -> np.ndarray | None:
    raw_color = text.get("background_rgb")
    if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
        return None
    try:
        color = np.asarray([int(round(float(v))) for v in raw_color], dtype=np.uint8)
    except Exception:
        return None
    luma = float(np.mean(color.astype(np.float32)))
    chroma = int(color.max()) - int(color.min())
    if luma >= 235.0 or luma <= 36.0:
        return color
    if chroma <= 4 and (luma >= 220.0 or luma <= 52.0):
        return color
    return None


def _text_geometry_mask(width: int, height: int, text: dict) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=np.uint8)
    has_polygon = False
    raw_polygons = text.get("line_polygons")
    if isinstance(raw_polygons, list):
        for polygon in raw_polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue
            points: list[list[int]] = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    px = max(0, min(width - 1, int(round(float(point[0])))))
                    py = max(0, min(height - 1, int(round(float(point[1])))))
                except Exception:
                    continue
                points.append([px, py])
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
                has_polygon = True

    if not has_polygon:
        bbox = (
            _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        if bbox is None:
            return None
        mask = _mask_from_bbox(width, height, bbox, padding=3)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if int(np.count_nonzero(mask)) < 24:
        return None
    return mask


def _try_metadata_background_text_fill(image_rgb: np.ndarray, text: dict) -> np.ndarray | None:
    if not _fast_metadata_background_fill_enabled():
        return None
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
    color = _metadata_background_color(text)
    if color is None:
        return None
    mask = _text_geometry_mask(width, height, text)
    if mask is None:
        return None
    mask_area = int(np.count_nonzero(mask))
    if mask_area > int(width * height * 0.35):
        return None

    bg_i = color.astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    ring = cv2.dilate(mask, kernel, iterations=1)
    ring = ((ring > 0) & (mask == 0))
    if int(np.count_nonzero(ring)) >= 32:
        ring_pixels = image_rgb[ring].astype(np.int16)
        ring_delta = np.mean(np.abs(ring_pixels - bg_i[None, :]), axis=1)
        if float(np.mean(ring_delta <= 28.0)) < 0.35:
            return None

    text_pixels = image_rgb[mask > 0].astype(np.int16)
    if text_pixels.size == 0:
        return None
    text_delta = np.mean(np.abs(text_pixels - bg_i[None, :]), axis=1)
    if float(np.percentile(text_delta, 90)) < 24.0:
        return None

    result = image_rgb.copy()
    result[mask > 0] = color
    return result


def _apply_fast_local_balloon_fill(
    band_rgb: np.ndarray,
    ocr_page: dict,
    vision_blocks: list[dict],
) -> tuple[np.ndarray, list[dict], dict]:
    rejection_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    def _record(stats: dict) -> dict:
        ocr_page["_strip_fast_local_balloon_count"] = stats["local_balloon_count"]
        ocr_page["_strip_remaining_inpaint_blocks"] = stats["remaining_blocks"]
        ocr_page["_strip_fast_local_rejection_reasons"] = dict(rejection_reasons)
        return stats

    fast_local_enabled = _fast_local_balloon_fill_enabled()
    if not fast_local_enabled or not vision_blocks:
        text_count = len([text for text in ocr_page.get("texts", []) if isinstance(text, dict)])
        rejection_reasons["disabled" if not fast_local_enabled else "no_vision_blocks"] = max(1, text_count)
        return band_rgb, vision_blocks, _record({"local_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    from vision_stack.runtime import _try_koharu_balloon_fill

    height, width = band_rgb.shape[:2]
    result = band_rgb.copy()
    filled_bboxes: list[list[int]] = []
    filled_keys: set[tuple[int, int, int, int]] = set()

    for text in ocr_page.get("texts", []):
        rejection_reason = _fast_local_rejection_reason(text)
        if rejection_reason:
            _reject(rejection_reason)
            continue
        text_bbox = (
            _normalize_bbox(text.get("text_pixel_bbox"), width, height)
            or _normalize_bbox(text.get("bbox"), width, height)
        )
        fill_bbox = _normalize_bbox(text.get("balloon_bbox"), width, height)
        if text_bbox is None:
            _reject("missing_text_bbox")
            continue
        if fill_bbox is None:
            _reject("missing_balloon_bbox")
            continue
        fill_key = tuple(fill_bbox)
        if fill_key in filled_keys:
            continue
        if not any(_block_is_covered_by_fast_fill(block, [fill_bbox], width, height) for block in vision_blocks):
            _reject("no_covered_vision_block")
            continue

        mask = _mask_from_bbox(width, height, text_bbox)
        filled = _try_koharu_balloon_fill(result, mask)
        if filled is None:
            filled = _try_solid_background_text_fill(result, text_bbox, fill_bbox)
        if filled is None:
            filled = _try_metadata_background_text_fill(result, text)
        if filled is None:
            _reject("no_flat_fill")
            continue

        result = filled
        filled_bboxes.append(fill_bbox)
        filled_keys.add(fill_key)

    if not filled_bboxes:
        return band_rgb, vision_blocks, _record({"local_balloon_count": 0, "remaining_blocks": len(vision_blocks)})

    remaining_blocks = [
        block
        for block in vision_blocks
        if not _block_is_covered_by_fast_fill(block, filled_bboxes, width, height)
    ]
    stats = {
        "local_balloon_count": len(filled_bboxes),
        "remaining_blocks": len(remaining_blocks),
    }
    return result, remaining_blocks, _record(stats)


def prewarm_band_inpainter(profile: str = "quality"):
    """Carrega o inpainter pesado cedo para sobrepor inicializacao com OCR."""
    from vision_stack.runtime import _get_inpainter

    return _get_inpainter(profile)


def inpaint_band_image(band_rgb: np.ndarray, ocr_page: dict) -> np.ndarray:
    """Aplica o mesmo round de inpaint do runtime principal na banda do strip."""
    from vision_stack.runtime import _apply_inpainting_round, _apply_post_inpaint_cleanup_timed, _get_inpainter

    if band_rgb.size == 0 or not ocr_page.get("texts"):
        return band_rgb.copy()

    height, width = band_rgb.shape[:2]
    vision_blocks = list(ocr_page.get("_vision_blocks") or [])
    if not vision_blocks:
        vision_blocks = _build_fallback_vision_blocks(ocr_page, width, height)
    if not vision_blocks:
        return band_rgb.copy()

    ocr_page["_strip_used_fast_white_fill"] = False
    ocr_page["_strip_used_fast_local_fill"] = False
    ocr_page["_strip_used_real_inpaint"] = False
    ocr_page["_strip_used_post_cleanup"] = False

    before_white = len(vision_blocks)
    working_rgb, vision_blocks, _ = _apply_fast_white_balloon_fill(
        band_rgb,
        ocr_page,
        vision_blocks,
    )
    if len(vision_blocks) != before_white:
        ocr_page["_strip_used_fast_white_fill"] = True

    before_local = len(vision_blocks)
    working_rgb, vision_blocks, _ = _apply_fast_local_balloon_fill(
        working_rgb,
        ocr_page,
        vision_blocks,
    )
    if len(vision_blocks) != before_local:
        ocr_page["_strip_used_fast_local_fill"] = True

    if not vision_blocks:
        if not _fast_white_post_cleanup_enabled():
            return working_rgb.copy()
        cleaned, cleanup_stats = _apply_post_inpaint_cleanup_timed(
            band_rgb,
            working_rgb,
            list(ocr_page.get("texts", [])),
        )
        ocr_page.update(cleanup_stats)
        ocr_page["_strip_used_post_cleanup"] = True
        return cleaned

    inpaint_payload = dict(ocr_page)
    inpaint_payload["_vision_blocks"] = vision_blocks
    inpaint_payload["_skip_internal_post_cleanup"] = True
    inpainter = _get_inpainter("quality")
    started = time.perf_counter()
    cleaned = _apply_inpainting_round(working_rgb, inpaint_payload, inpainter)
    ocr_page["_t_lama_total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    round_stats = inpaint_payload.get("_inpaint_round_stats")
    if isinstance(round_stats, dict):
        ocr_page.update(round_stats)
    ocr_page["_strip_used_real_inpaint"] = True
    cleaned, cleanup_stats = _apply_post_inpaint_cleanup_timed(
        band_rgb,
        cleaned,
        list(ocr_page.get("texts", [])),
    )
    ocr_page.update(cleanup_stats)
    ocr_page["_strip_used_post_cleanup"] = True
    return cleaned.copy() if cleaned is working_rgb else cleaned
