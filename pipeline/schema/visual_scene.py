"""Page-space text identities for the R1 shadow reconciliation contract."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _pages(project_data: dict[str, Any]) -> list[dict[str, Any]]:
    pages = project_data.get("paginas") or project_data.get("pages") or []
    return [page for page in pages if isinstance(page, dict)]


def _bbox(layer: dict[str, Any]) -> tuple[int, int, int, int] | None:
    raw = layer.get("text_pixel_bbox") or layer.get("bbox") or layer.get("source_bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in raw[:4])
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _area(bbox: tuple[int, int, int, int] | None) -> int:
    return 0 if bbox is None else max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _band_bbox(layer: dict[str, Any]) -> tuple[int, int, int, int] | None:
    raw = layer.get("band_bbox") or layer.get("_band_bbox")
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        try:
            x1, y1, x2, y2 = (int(round(float(value))) for value in raw[:4])
        except (TypeError, ValueError):
            return None
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
    return None


def _coverage_and_edge_distance(
    layer: dict[str, Any],
) -> tuple[float, int]:
    text_bbox, band_bbox = _bbox(layer), _band_bbox(layer)
    if text_bbox is None or band_bbox is None:
        return (0.0, 0)
    x1, y1 = max(text_bbox[0], band_bbox[0]), max(text_bbox[1], band_bbox[1])
    x2, y2 = min(text_bbox[2], band_bbox[2]), min(text_bbox[3], band_bbox[3])
    coverage = max(0, x2 - x1) * max(0, y2 - y1) / max(1, _area(text_bbox))
    edge_distance = max(
        0,
        min(
            text_bbox[0] - band_bbox[0],
            text_bbox[1] - band_bbox[1],
            band_bbox[2] - text_bbox[2],
            band_bbox[3] - text_bbox[3],
        ),
    )
    return (round(coverage, 4), edge_distance)


def _iou(left: tuple[int, int, int, int] | None, right: tuple[int, int, int, int] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union else 0.0


def _text(layer: dict[str, Any]) -> str:
    return str(layer.get("translated") or layer.get("traduzido") or layer.get("text") or layer.get("original") or "").strip()


def _normalize_text(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _identity_tokens(layer: dict[str, Any]) -> set[str]:
    """Return page-stable identity evidence, never local OCR labels.

    Labels such as ``ocr_001`` are intentionally reused by every band.  They
    become valid ownership evidence only once qualified as ``trace@band``.
    """
    tokens: set[str] = set()
    trace_id = str(layer.get("trace_id") or "").strip()
    if "@" in trace_id:
        tokens.add(trace_id)
    for key in ("source_trace_ids", "_source_trace_ids"):
        tokens.update(
            value
            for value in (str(item).strip() for item in layer.get(key) or [])
            if "@" in value
        )
    return tokens


def _band_id(layer: dict[str, Any]) -> str:
    explicit = str(layer.get("band_id") or "").strip()
    if explicit:
        return explicit
    trace = str(layer.get("trace_id") or "")
    return trace.split("@", 1)[1] if "@" in trace else ""


def _page_id(page: dict[str, Any], index: int) -> str:
    return str(page.get("id") or page.get("page_id") or f"page_{index:03d}")


def _explicitly_suppressed(layer: dict[str, Any]) -> bool:
    policy = str(layer.get("render_policy") or layer.get("route_action") or "").lower()
    flags = {str(flag).lower() for flag in layer.get("qa_flags") or []}
    return "suppressed" in policy or "merged_into_primary" in policy or any("suppressed" in flag for flag in flags)


def _lifecycle(layer: dict[str, Any]) -> str:
    if _explicitly_suppressed(layer):
        return "suppressed"
    if layer.get("visible", True) is not False and (
        bool(layer.get("rendered"))
        or isinstance(layer.get("render_bbox"), (list, tuple))
        or isinstance(layer.get("safe_text_box"), (list, tuple))
    ):
        return "rendered"
    if bool(layer.get("ocr_accepted")):
        return "ocr_accepted"
    return "detected"


def _group_indices(layers: list[dict[str, Any]]) -> list[list[int]]:
    parents = list(range(len(layers)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    tokens = [_identity_tokens(layer) for layer in layers]
    bboxes = [_bbox(layer) for layer in layers]
    for left in range(len(layers)):
        for right in range(left + 1, len(layers)):
            if tokens[left].intersection(tokens[right]):
                union(left, right)
                continue
            left_text, right_text = _normalize_text(_text(layers[left])), _normalize_text(_text(layers[right]))
            same_or_fragment = bool(left_text and right_text and (left_text in right_text or right_text in left_text))
            if same_or_fragment and _iou(bboxes[left], bboxes[right]) >= 0.55:
                union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(layers)):
        groups[find(index)].append(index)
    return list(groups.values())


def build_visual_text_instances(project_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build deterministic shadow identities without changing project layers."""
    instances: list[dict[str, Any]] = []
    for page_index, page in enumerate(_pages(project_data), start=1):
        page_id = _page_id(page, page_index)
        layers = [layer for layer in page.get("text_layers") or [] if isinstance(layer, dict)]
        containers: dict[tuple[int, int, int, int], str] = {}
        for layer in layers:
            raw_container = layer.get("balloon_bbox") or layer.get("bubble_bbox")
            if isinstance(raw_container, (list, tuple)) and len(raw_container) >= 4:
                try:
                    key = tuple(int(round(float(value))) for value in raw_container[:4])
                except (TypeError, ValueError):
                    continue
                containers.setdefault(key, f"{page_id}_container_{len(containers) + 1:03d}")
        groups = _group_indices(layers)
        staged: list[dict[str, Any]] = []
        for member_indices in groups:
            members = [layers[index] for index in member_indices]
            def owner_key(layer: dict[str, Any]) -> tuple[Any, ...]:
                coverage, edge_distance = _coverage_and_edge_distance(layer)
                return (
                    _explicitly_suppressed(layer),
                    -coverage,
                    -edge_distance,
                    -len(_normalize_text(_text(layer))),
                    -_area(_bbox(layer)),
                    _band_id(layer),
                    str(layer.get("id") or layer.get("trace_id") or ""),
                )

            owner = sorted(members, key=owner_key)[0]
            coverage_ratio, edge_distance = _coverage_and_edge_distance(owner)
            all_bands = sorted({band for layer in members for band in [_band_id(layer)] if band})
            all_tokens = sorted({token for layer in members for token in _identity_tokens(layer) if "@" in token})
            reading_order = min(int(layer.get("reading_order") or 10**9) for layer in members)
            raw_container = owner.get("balloon_bbox") or owner.get("bubble_bbox")
            container_id = None
            if isinstance(raw_container, (list, tuple)) and len(raw_container) >= 4:
                try:
                    container_id = containers.get(tuple(int(round(float(value))) for value in raw_container[:4]))
                except (TypeError, ValueError):
                    container_id = None
            staged.append(
                {
                    "bbox": list(_bbox(owner) or (0, 0, 0, 0)),
                    "container_id": container_id or f"{page_id}_container_unassigned",
                    "lifecycle": _lifecycle(owner),
                    "observer_band_ids": all_bands,
                    "owner_band_id": _band_id(owner),
                    "owner_layer_id": str(owner.get("id") or owner.get("text_id") or owner.get("trace_id") or ""),
                    "owner_selection": {
                        "coverage_ratio": coverage_ratio,
                        "edge_distance": edge_distance,
                        "reason": "full_band_coverage_then_edge_distance",
                    },
                    "owner_qa_flags": [str(flag) for flag in owner.get("qa_flags") or [] if str(flag)],
                    "owner_text": _text(owner),
                    "page_id": page_id,
                    "reading_order": reading_order,
                    "source_trace_ids": all_tokens,
                    "layer_ids": [str(layer.get("id") or layer.get("text_id") or layer.get("trace_id") or "") for layer in members],
                }
            )
        staged.sort(key=lambda item: (item["reading_order"], item["bbox"][1], item["bbox"][0], item["owner_layer_id"]))
        for ordinal, instance in enumerate(staged, start=1):
            instance["component_id"] = f"{page_id}_component_{ordinal:03d}"
            instance["text_instance_id"] = f"{page_id}_text_{ordinal:03d}"
            instance["orientation_deg"] = 0.0
            instances.append(instance)
    return instances
