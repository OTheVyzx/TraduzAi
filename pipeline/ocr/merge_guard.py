"""Shared provenance checks for OCR merge decisions.

These helpers deliberately describe geometry and source provenance only.  They
do not decide whether a record is dialogue, SFX, or eligible for rendering.
"""

from __future__ import annotations

from typing import Any


RISKY_CROP_FALLBACK_REASON = "low_confidence_crop_fallback_dominates_geometry"


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _area(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _assignment_mode(record: dict[str, Any]) -> str:
    audit = record.get("_ocr_assignment_audit")
    if isinstance(audit, dict):
        mode = str(audit.get("assignment_mode") or "").strip()
        if mode:
            return mode
    return str(record.get("ocr_assignment_mode") or "").strip()


def _confidence(record: dict[str, Any]) -> float:
    try:
        return round(float(record.get("confidence", record.get("ocr_confidence", 0.0)) or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def source_provenance(record: dict[str, Any]) -> dict[str, Any]:
    bbox = _bbox4(record.get("text_pixel_bbox")) or _bbox4(record.get("bbox"))
    return {
        "text_id": str(record.get("text_id") or record.get("id") or ""),
        "trace_id": str(record.get("trace_id") or ""),
        "text": str(
            record.get("normalized_text_final")
            or record.get("normalized_ocr")
            or record.get("text")
            or record.get("raw_ocr")
            or record.get("original")
            or ""
        ).strip(),
        "bbox": list(bbox) if bbox is not None else None,
        "area": _area(bbox),
        "confidence": _confidence(record),
        "assignment_mode": _assignment_mode(record),
        "has_line_geometry": bool(record.get("line_polygons")),
    }


def build_merge_risk_audit(
    records: list[dict[str, Any]],
    *,
    merge_stage: str,
    merge_path: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic audit for a prospective OCR merge."""

    sources = [source_provenance(record) for record in records if isinstance(record, dict)]
    trusted = [
        source
        for source in sources
        if source["assignment_mode"] == "full_page_lines"
        and source["confidence"] >= 0.80
        and source["has_line_geometry"]
    ]
    weak_crop = [
        source
        for source in sources
        if source["assignment_mode"] == "crop_fallback"
        and source["confidence"] <= 0.65
        and not source["has_line_geometry"]
    ]
    suspicious_pairs: list[dict[str, Any]] = []
    for strong in trusted:
        for weak in weak_crop:
            ratio = float(weak["area"]) / float(max(1, strong["area"]))
            if ratio >= 2.5:
                suspicious_pairs.append(
                    {
                        "trusted_text_id": strong["text_id"],
                        "weak_text_id": weak["text_id"],
                        "weak_to_trusted_area_ratio": round(ratio, 3),
                    }
                )
    audit = {
        "merge_stage": merge_stage,
        "source_count": len(sources),
        "sources": sources,
        "suspicious": bool(suspicious_pairs),
        "reason": RISKY_CROP_FALLBACK_REASON if suspicious_pairs else "no_risky_crop_fallback_pattern",
        "suspicious_pairs": suspicious_pairs,
    }
    if merge_path:
        audit["merge_path"] = merge_path
    return audit


def weak_source_ids_from_risk_audit(audit: dict[str, Any] | None) -> set[str]:
    if not isinstance(audit, dict):
        return set()
    return {
        str(pair.get("weak_text_id") or "").strip()
        for pair in audit.get("suspicious_pairs") or []
        if isinstance(pair, dict) and str(pair.get("weak_text_id") or "").strip()
    }
