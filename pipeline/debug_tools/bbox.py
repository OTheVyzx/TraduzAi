from __future__ import annotations

import re
from typing import Any, Iterable


BBOX_KEYS = (
    "source_bbox",
    "bbox",
    "text_pixel_bbox",
    "balloon_bbox",
    "bubble_mask_bbox",
    "bubble_inner_bbox",
    "balloon_inner_bbox",
    "layout_bbox",
    "render_bbox",
    "safe_text_box",
    "_debug_safe_text_box",
    "layout_safe_bbox",
    "position_bbox",
    "capacity_bbox",
    "target_bbox",
    "connected_position_bboxes",
)


def _bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(value: Any) -> int:
    bbox = _bbox(value)
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _get_page_attr(page: Any, key: str, default: Any = None) -> Any:
    if isinstance(page, dict):
        return page.get(key, default)
    return getattr(page, key, default)


def _page_texts(page: Any) -> list[dict[str, Any]]:
    if isinstance(page, dict):
        texts = page.get("texts")
        if texts is None and isinstance(page.get("text_layers"), dict):
            texts = page["text_layers"].get("texts")
    else:
        text_layers = getattr(page, "text_layers", None)
        texts = text_layers.get("texts") if isinstance(text_layers, dict) else None
    return [text for text in list(texts or []) if isinstance(text, dict)]


def _page_height(page: Any) -> int:
    explicit = _get_page_attr(page, "height", None)
    if explicit:
        return int(explicit)
    image = _get_page_attr(page, "image", None)
    shape = getattr(image, "shape", None)
    if shape and len(shape) >= 2:
        return int(shape[0])
    y_top = int(_get_page_attr(page, "y_top", 0) or 0)
    y_bottom = int(_get_page_attr(page, "y_bottom", 0) or 0)
    return max(0, y_bottom - y_top)


def _page_width(page: Any) -> int:
    explicit = _get_page_attr(page, "width", None)
    if explicit:
        return int(explicit)
    image = _get_page_attr(page, "image", None)
    shape = getattr(image, "shape", None)
    if shape and len(shape) >= 2:
        return int(shape[1])
    return 0


def _text_id(text: dict[str, Any], index: int) -> str:
    return str(text.get("id") or text.get("text_id") or f"ocr_{index:03d}")


def _page_id(page: Any, index: int) -> str:
    return str(_get_page_attr(page, "page_id", None) or _get_page_attr(page, "id", None) or f"page_{index:03d}")


def _page_id_from_band_id(value: Any) -> str | None:
    if not value:
        return None
    match = re.match(r"^(page_\d{3})_band_\d{3}$", str(value))
    return match.group(1) if match else None


def _collect_nested_bbox_values(prefix: str, value: Any) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    if not isinstance(value, dict):
        return collected
    for key, item in value.items():
        key_str = str(key)
        path = f"{prefix}.{key_str}"
        if key_str.endswith("bbox") or key_str.endswith("_bbox") or key_str.endswith("bboxes"):
            if key_str.endswith("bboxes") and isinstance(item, list):
                bboxes = [bbox for bbox in (_bbox(candidate) for candidate in item) if bbox is not None]
                if bboxes:
                    collected[path] = bboxes
                continue
            bbox = _bbox(item)
            if bbox is not None:
                collected[path] = bbox
                continue
        collected.update(_collect_nested_bbox_values(path, item))
    return collected


def layout_block_records(pages: Iterable[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page_index, page in enumerate(list(pages or []), start=1):
        fallback_page_id = _page_id(page, page_index)
        page_height = _page_height(page)
        page_width = _page_width(page)
        for text_index, text in enumerate(_page_texts(page), start=1):
            band_id = str(text.get("band_id") or "")
            text_page_id = str(
                text.get("page_id")
                or _page_id_from_band_id(band_id)
                or fallback_page_id
            )
            bboxes = {}
            for key in BBOX_KEYS:
                if key.endswith("bboxes"):
                    value = [bbox for bbox in (_bbox(item) for item in list(text.get(key) or [])) if bbox is not None]
                else:
                    value = _bbox(text.get(key))
                bboxes[key] = {"value": value or [], "space": "page"}
            for nested_key in ("qa_metrics", "_render_debug"):
                for key, value in _collect_nested_bbox_values(nested_key, text.get(nested_key)).items():
                    bboxes[key] = {"value": value or [], "space": "page"}
            records.append(
                {
                    "schema_version": 1,
                    "text_id": _text_id(text, text_index),
                    "page_id": text_page_id,
                    "band_id": band_id,
                    "coordinate_space": "page",
                    "source_coordinate_space": str(text.get("source_coordinate_space") or ""),
                    "band_y_top": int(text.get("band_y_top") or text.get("_band_y_top") or 0),
                    "band_height": int(text.get("band_height") or text.get("_band_height") or 0),
                    "page_height": page_height,
                    "page_width": page_width,
                    "bboxes": bboxes,
                    "polygons": {
                        "line_polygons_count": len(text.get("line_polygons") or []),
                        "balloon_polygon_present": bool(text.get("balloon_polygon")),
                    },
                    "decision_trace_reason": str(text.get("decision_trace_reason") or text.get("layout_reason") or ""),
                }
            )
    return records


def _bbox_value(record: dict[str, Any], key: str) -> list[int] | None:
    value = ((record.get("bboxes") or {}).get(key) or {}).get("value")
    return _bbox(value)


def _bbox_values(record: dict[str, Any], key: str) -> list[list[int]]:
    value = ((record.get("bboxes") or {}).get(key) or {}).get("value")
    if isinstance(value, list) and value and all(isinstance(item, (list, tuple)) for item in value):
        return [bbox for bbox in (_bbox(item) for item in value) if bbox is not None]
    bbox = _bbox(value)
    return [bbox] if bbox is not None else []


def audit_bbox_coordinate_space(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records_list = list(records or [])
    findings: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, int]] = {}
    for record in records_list:
        page_height = int(record.get("page_height") or 0)
        band_y_top = int(record.get("band_y_top") or 0)
        band_height = max(1, int(record.get("band_height") or 0))
        reference_global_y = max(
            (
                bbox[1]
                for key in ("source_bbox", "balloon_bbox", "text_pixel_bbox", "bbox", "layout_bbox")
                for bbox in _bbox_values(record, key)
            ),
            default=0,
        )
        for key in sorted((record.get("bboxes") or {}).keys()):
            values = _bbox_values(record, key)
            if not values:
                continue
            key_summary = by_key.setdefault(key, {"page": 0, "band": 0, "mismatch": 0})
            key_has_band_local = False
            for value in values:
                appears_band_local = bool(
                    band_y_top >= max(400, band_height)
                    and value[1] < max(400, band_height)
                    and (
                        reference_global_y >= band_y_top
                        or page_height > 4000
                        or value[1] + band_y_top < page_height
                    )
                )
                if appears_band_local:
                    key_summary["band"] += 1
                    key_has_band_local = True
                else:
                    key_summary["page"] += 1
            if key_has_band_local:
                key_summary["mismatch"] += 1
                findings.append(
                    {
                        "text_id": record.get("text_id", ""),
                        "page_id": record.get("page_id", ""),
                        "band_id": record.get("band_id", ""),
                        "key": key,
                        "value": values[0],
                        "expected_space": "page",
                        "band_y_top": band_y_top,
                        "issue": "derived_bbox_appears_band_local",
                        "severity": "critical",
                        "blocker": "derived_bbox_coordinate_mismatch",
                    }
                )
        if page_height <= 4000:
            continue
        bbox = _bbox_value(record, "bbox")
        source_bbox = _bbox_value(record, "source_bbox")
        text_pixel_bbox = _bbox_value(record, "text_pixel_bbox")
        layout_bbox = _bbox_value(record, "layout_bbox")
        threshold = band_height + int(band_y_top * 0.5)
        y_values = [
            candidate[1]
            for candidate in (bbox, source_bbox, text_pixel_bbox, layout_bbox, _bbox_value(record, "balloon_bbox"))
            if candidate is not None
        ]
        max_delta = max(y_values) - min(y_values) if len(y_values) >= 2 else 0
        band_local_in_page = bool(
            bbox
            and source_bbox
            and bbox[1] < page_height * 0.2
            and source_bbox[1] > max(1, bbox[1]) * 5
        )
        if max_delta <= threshold and not band_local_in_page:
            continue
        findings.append(
            {
                "text_id": record.get("text_id", ""),
                "page_id": record.get("page_id", ""),
                "band_id": record.get("band_id", ""),
                "band_y_top": band_y_top,
                "issue": "bbox_appears_band_local" if band_local_in_page else "mixed_bbox_coordinate_space",
                "evidence": {
                    "bbox_y_top": bbox[1] if bbox else None,
                    "source_bbox_y_top": source_bbox[1] if source_bbox else None,
                    "text_pixel_bbox_y_top": text_pixel_bbox[1] if text_pixel_bbox else None,
                    "layout_bbox_y_top": layout_bbox[1] if layout_bbox else None,
                    "delta": max_delta,
                },
                "severity": "critical",
                "blocker": "layout_bbox_coordinate_mismatch",
            }
        )
    total = len(records_list)
    mixed_findings = [item for item in findings if item.get("blocker") == "layout_bbox_coordinate_mismatch"]
    derived_findings = [item for item in findings if item.get("blocker") == "derived_bbox_coordinate_mismatch"]
    return {
        "schema_version": 1,
        "summary": {
            "total_text_layers": total,
            "all_consistent": not findings,
            "mixed_coordinate_space_count": len(mixed_findings),
            "derived_bbox_coordinate_mismatch_count": len(derived_findings),
            "by_key": by_key,
            "band_local_in_page_context_count": sum(1 for item in mixed_findings if item["issue"] == "bbox_appears_band_local"),
            "page_global_in_band_context_count": 0,
        },
        "findings": findings,
    }


def coordinate_audit_flags(audit: dict[str, Any]) -> list[str]:
    findings = list(audit.get("findings") or [])
    flags: set[str] = set()
    if any(item.get("blocker") == "layout_bbox_coordinate_mismatch" for item in findings):
        flags.add("layout_bbox_coordinate_mismatch")
    derived_mismatches = [
        item for item in findings if item.get("blocker") == "derived_bbox_coordinate_mismatch"
    ]
    if derived_mismatches:
        flags.add("layout_bbox_coordinate_mismatch")
    if any(
        str(item.get("key") or "") in {"bubble_inner_bbox", "bubble_mask_bbox", "balloon_inner_bbox"}
        for item in derived_mismatches
    ):
        flags.add("bubble_inner_bbox_coordinate_mismatch")
    if any(
        str(item.get("key") or "") in {"bubble_inner_bbox", "bubble_mask_bbox", "balloon_inner_bbox", "safe_text_box", "render_bbox"}
        for item in derived_mismatches
    ):
        flags.add("page_space_rerender_mixed_coordinates")
    return sorted(flags)


def source_bbox_balloon_overreach(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for record in records or []:
        source_bbox = _bbox_value(record, "source_bbox")
        balloon_bbox = _bbox_value(record, "balloon_bbox")
        text_pixel_bbox = _bbox_value(record, "text_pixel_bbox")
        if not source_bbox or source_bbox != balloon_bbox:
            continue
        source_area = _bbox_area(source_bbox)
        text_pixel_area = _bbox_area(text_pixel_bbox)
        if text_pixel_area <= 0:
            continue
        ratio = source_area / float(text_pixel_area)
        critical = ratio >= 4.0
        findings.append(
            {
                "schema_version": 1,
                "text_id": record.get("text_id", ""),
                "page_id": record.get("page_id", ""),
                "band_id": record.get("band_id", ""),
                "issue": "source_bbox_equals_balloon_bbox",
                "source_bbox": source_bbox,
                "balloon_bbox": balloon_bbox,
                "text_pixel_bbox": text_pixel_bbox or [],
                "source_area": source_area,
                "text_pixel_area": text_pixel_area,
                "area_ratio": round(ratio, 2),
                "decision_trace_reason": record.get("decision_trace_reason", ""),
                "severity": "critical" if critical else "warning",
                "blocker": "source_bbox_assigned_from_balloon",
            }
        )
    return findings


def write_layout_geometry_debug_artifacts(pages: Iterable[Any], recorder: Any | None = None) -> None:
    if recorder is None:
        try:
            from debug_tools import get_recorder
        except Exception:
            return
        recorder = get_recorder()
    if not recorder or not getattr(recorder, "enabled", False):
        return
    records = layout_block_records(pages)
    for record in records:
        recorder.write_jsonl("05_layout_geometry/layout_blocks.jsonl", record)
    recorder.write_json("05_layout_geometry/bbox_coordinate_audit.json", audit_bbox_coordinate_space(records))
    for finding in source_bbox_balloon_overreach(records):
        recorder.write_jsonl("05_layout_geometry/source_bbox_balloon_overreach.jsonl", finding)
