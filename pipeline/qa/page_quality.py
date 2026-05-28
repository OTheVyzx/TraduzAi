"""Page-level OCR quality routing for adaptive CJK reruns."""

from __future__ import annotations

import re
from typing import Any


SOURCE_SCRIPT_RE = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)

OCR_RERUN_FLAGS = {
    "ocr_run_on_suspect",
    "partial_ocr",
    "ocr_partial_multiline",
    "ocr_merge_suspect",
    "missing_text_in_speech_balloon",
}

NON_RERUN_FLAGS = {
    "vlm_failure_phrase",
    "translation_fallback_phrase",
    "literal_ocr_translation",
    "sfx_preserved",
    "sfx_candidate",
    "fill_normalization",
    "typesetting",
    "text_overflow",
}


def is_cjk_source(source_lang: str | None) -> bool:
    return str(source_lang or "").strip().lower() in {"ja", "jp", "ko", "kr", "zh", "zh-cn", "zh-tw"}


def evaluate_page_quality(
    page_result: dict[str, Any],
    *,
    source_lang: str = "en",
    chapter_prior: dict[str, Any] | None = None,
    expanded_reocr_attempted: bool = False,
) -> dict[str, Any]:
    """Return cheap routing signals without running any expensive fallback."""

    texts = [item for item in page_result.get("texts", []) or [] if isinstance(item, dict)]
    blocks = [item for item in page_result.get("_vision_blocks", []) or [] if isinstance(item, dict)]
    cjk = is_cjk_source(source_lang)
    issues: list[dict[str, Any]] = []
    non_rerun_issues: list[dict[str, Any]] = []

    for index, text in enumerate(texts):
        flags = {str(flag) for flag in text.get("qa_flags") or [] if flag}
        bbox = text.get("bbox") or text.get("text_pixel_bbox")
        confidence = _as_float(text.get("confidence"), 1.0)
        raw_text = str(text.get("text") or text.get("original") or "")
        line_count = _estimated_line_count(text)

        if flags & OCR_RERUN_FLAGS:
            issues.append(_issue("ocr_flagged_for_rerun", index, bbox, flags=sorted(flags & OCR_RERUN_FLAGS)))
        if cjk and line_count >= 2 and _looks_partially_cjk(raw_text, confidence):
            issues.append(_issue("partial_multiline_ocr", index, bbox, confidence=confidence))
        if cjk and _looks_like_merged_cjk(raw_text, bbox):
            issues.append(_issue("probable_cjk_merge", index, bbox, text=raw_text[:80]))
        if flags & NON_RERUN_FLAGS:
            non_rerun_issues.append(
                _issue("non_rerun_quality_flag", index, bbox, flags=sorted(flags & NON_RERUN_FLAGS))
            )

    missing_block_count = _count_known_balloons_without_text(blocks, texts)
    if missing_block_count:
        issues.append(
            {
                "type": "known_speech_balloon_without_ocr",
                "severity": "high",
                "count": missing_block_count,
            }
        )

    prior_ratio = _text_count_ratio(texts, chapter_prior)
    if prior_ratio is not None and prior_ratio < 0.55:
        issues.append(
            {
                "type": "low_ocr_coverage_vs_chapter_prior",
                "severity": "high",
                "text_count_ratio": prior_ratio,
            }
        )

    should_try_bbox = bool(cjk and issues)
    page_detect_reasons = [
        issue
        for issue in issues
        if issue["type"]
        in {
            "partial_multiline_ocr",
            "probable_cjk_merge",
            "known_speech_balloon_without_ocr",
            "low_ocr_coverage_vs_chapter_prior",
        }
    ]
    should_try_page_detect = bool(cjk and expanded_reocr_attempted and page_detect_reasons)

    return {
        "source_lang": source_lang,
        "mode": "cjk_adaptive" if cjk else "standard",
        "text_count": len(texts),
        "vision_block_count": len(blocks),
        "issues": _dedupe_issues(issues),
        "non_rerun_issues": _dedupe_issues(non_rerun_issues),
        "text_count_ratio": prior_ratio,
        "should_try_bbox_expanded_reocr": should_try_bbox,
        "should_try_page_detect": should_try_page_detect,
        "page_detect_auto_allowed": False,
    }


def _issue(issue_type: str, index: int, bbox, **extra: Any) -> dict[str, Any]:
    payload = {
        "type": issue_type,
        "severity": "high",
        "text_index": index,
    }
    if bbox:
        payload["bbox"] = list(bbox)
    payload.update(extra)
    return payload


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        key = (
            issue.get("type"),
            issue.get("text_index"),
            tuple(issue.get("bbox") or []),
            tuple(issue.get("flags") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _estimated_line_count(text: dict[str, Any]) -> int:
    polygons = text.get("line_polygons")
    if isinstance(polygons, list) and polygons:
        return len(polygons)
    raw = str(text.get("text") or text.get("original") or "")
    return max(1, raw.count("\n") + 1)


def _looks_partially_cjk(text: str, confidence: float) -> bool:
    if confidence >= 0.62:
        return False
    cjk_count = len(SOURCE_SCRIPT_RE.findall(text or ""))
    if cjk_count == 0:
        return False
    alnum_count = sum(1 for ch in text if ch.isalnum())
    return cjk_count <= max(2, int(alnum_count * 0.45))


def _looks_like_merged_cjk(text: str, bbox) -> bool:
    cjk_count = len(SOURCE_SCRIPT_RE.findall(text or ""))
    if cjk_count < 2:
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    aspect = width / float(height)
    return aspect >= 6.0 and cjk_count <= 4


def _count_known_balloons_without_text(blocks: list[dict[str, Any]], texts: list[dict[str, Any]]) -> int:
    missing = 0
    for block in blocks:
        bbox = block.get("balloon_bbox") or block.get("bbox")
        if not bbox or not _is_speech_balloon_block(block):
            continue
        if not any(_bbox_matches_text(bbox, text.get("bbox") or text.get("text_pixel_bbox")) for text in texts):
            missing += 1
    return missing


def _is_speech_balloon_block(block: dict[str, Any]) -> bool:
    if block.get("balloon_polygon") or block.get("balloon_subregions") or block.get("connected_lobe_bboxes"):
        return True
    detector = str(block.get("detector") or "")
    return "balloon" in detector


def _bbox_matches_text(block_bbox, text_bbox) -> bool:
    if not block_bbox or not text_bbox:
        return False
    try:
        bx1, by1, bx2, by2 = [float(v) for v in block_bbox[:4]]
        tx1, ty1, tx2, ty2 = [float(v) for v in text_bbox[:4]]
    except Exception:
        return False
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    if bx1 - 8 <= cx <= bx2 + 8 and by1 - 8 <= cy <= by2 + 8:
        return True
    ix1, iy1 = max(bx1, tx1), max(by1, ty1)
    ix2, iy2 = min(bx2, tx2), min(by2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    text_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
    return ((ix2 - ix1) * (iy2 - iy1)) / text_area >= 0.20


def _text_count_ratio(texts: list[dict[str, Any]], chapter_prior: dict[str, Any] | None) -> float | None:
    if not chapter_prior:
        return None
    expected = chapter_prior.get("expected_text_count") or chapter_prior.get("median_text_count")
    try:
        expected_count = float(expected)
    except Exception:
        return None
    if expected_count <= 0:
        return None
    return len(texts) / expected_count
