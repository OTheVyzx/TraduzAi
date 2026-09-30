"""Independent, pre-inpaint visual container recovery for owner layout."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def _bbox(value: Any, *, width: int, height: int) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("container recovery requires a canonical bbox")
    result = tuple(int(item) for item in value)
    if (
        result[0] < 0
        or result[1] < 0
        or result[2] <= result[0]
        or result[3] <= result[1]
        or result[2] > width
        or result[3] > height
    ):
        raise ValueError("container recovery bbox is outside the logical page")
    return result


def _contains(outer: BBox, inner: BBox) -> bool:
    return bool(
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _proportional_fallback(seed: BBox, *, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = seed
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = max(12, int(box_width * 0.22))
    pad_y = max(12, int(box_height * 0.30))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def _threshold_visual_container(
    image_rgb: np.ndarray,
    *,
    semantic_bbox: BBox,
) -> dict[str, Any] | None:
    """Return a page-global light/dark region enclosing the text support.

    Candidate ranking is intentionally independent of the calling component.
    Multiple OCR lines surrounded by the same visual region therefore resolve
    to the same bbox instead of receiving line-local proportional containers.
    """

    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2GRAY)
    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thresholds = sorted(
        {
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            144,
            160,
            176,
            192,
            208,
            224,
            240,
            int(otsu_threshold),
        }
    )
    sx1, sy1, sx2, sy2 = semantic_bbox
    support_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    page_area = width * height
    candidates: list[dict[str, Any]] = []

    for threshold in thresholds:
        for polarity in ("bright", "dark"):
            binary = (
                gray >= threshold if polarity == "bright" else gray <= threshold
            ).astype(np.uint8)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
            support_labels, support_counts = np.unique(
                labels[sy1:sy2, sx1:sx2],
                return_counts=True,
            )
            for label, support_count in zip(support_labels, support_counts):
                label = int(label)
                if label == 0 or label >= count:
                    continue
                support_fraction = float(support_count) / float(support_area)
                if support_fraction < 0.32:
                    continue
                x, y, box_width, box_height, component_area = (
                    int(value) for value in stats[label]
                )
                bbox = (x, y, x + box_width, y + box_height)
                if not _contains(bbox, semantic_bbox):
                    continue
                bbox_area = box_width * box_height
                if bbox_area >= int(page_area * 0.96):
                    continue
                if box_width < (sx2 - sx1) + 4 or box_height < (sy2 - sy1) + 4:
                    continue
                if component_area < int(support_area * 1.25):
                    continue
                fill_ratio = float(component_area) / float(max(1, bbox_area))
                if fill_ratio < 0.15:
                    continue
                touches = sum(
                    (
                        x == 0,
                        y == 0,
                        x + box_width == width,
                        y + box_height == height,
                    )
                )
                if touches >= 3:
                    continue
                candidates.append(
                    {
                        "bbox_page": bbox,
                        "component_area": component_area,
                        "fill_ratio": fill_ratio,
                        "support_fraction": support_fraction,
                        "threshold": threshold,
                        "polarity": polarity,
                    }
                )

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            (int(item["bbox_page"][2]) - int(item["bbox_page"][0]))
            * (int(item["bbox_page"][3]) - int(item["bbox_page"][1])),
            -float(item["support_fraction"]),
            -float(item["fill_ratio"]),
            tuple(item["bbox_page"]),
            int(item["threshold"]),
            str(item["polarity"]),
        ),
    )


def _visual_container_evidence(
    image_rgb: np.ndarray,
    *,
    bbox_page: BBox,
    confidence: float,
    **details: Any,
) -> dict[str, Any]:
    page_pixels = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    payload = {
        "page_pixel_sha256": sha256(page_pixels.tobytes()).hexdigest(),
        "page_shape": list(page_pixels.shape),
        "source": "full_page_visual_container",
        "bbox_page": list(bbox_page),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "evidence_id": f"full_page_visual_container:{digest}",
        "source": "full_page_visual_container",
        "bbox_page": bbox_page,
        "page_shape": tuple(int(value) for value in page_pixels.shape),
        "confidence": round(float(confidence), 6),
        **details,
    }


def canonical_component_container_ids(
    evidence_by_component: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Canonicalize nested local evidence without joining adjacent containers."""

    component_ids = tuple(sorted(str(value) for value in evidence_by_component))
    parents = {component_id: component_id for component_id in component_ids}

    def find(component_id: str) -> str:
        current = component_id
        while parents[current] != current:
            parents[current] = parents[parents[current]]
            current = parents[current]
        return current

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    def containment_fraction(left: BBox, right: BBox) -> float:
        ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
        ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        smaller_area = min(
            max(1, (left[2] - left[0]) * (left[3] - left[1])),
            max(1, (right[2] - right[0]) * (right[3] - right[1])),
        )
        return float(intersection) / float(smaller_area)

    def is_page_spanning(item: dict[str, Any]) -> bool:
        bbox = tuple(int(value) for value in item.get("bbox_page") or ())
        shape = tuple(int(value) for value in item.get("page_shape") or ())
        if len(bbox) != 4 or len(shape) < 2:
            return False
        page_height, page_width = shape[:2]
        if page_width <= 0 or page_height <= 0:
            return False
        box_width = max(0, bbox[2] - bbox[0])
        box_height = max(0, bbox[3] - bbox[1])
        return (
            box_width * box_height >= page_width * page_height * 0.45
            or (box_width >= page_width * 0.85 and box_height >= page_height * 0.45)
        )

    def semantic_neighbours(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_bbox = tuple(
            int(value) for value in left.get("semantic_bbox_page") or ()
        )
        right_bbox = tuple(
            int(value) for value in right.get("semantic_bbox_page") or ()
        )
        if len(left_bbox) != 4 or len(right_bbox) != 4:
            return True
        left_width = max(1, left_bbox[2] - left_bbox[0])
        right_width = max(1, right_bbox[2] - right_bbox[0])
        left_height = max(1, left_bbox[3] - left_bbox[1])
        right_height = max(1, right_bbox[3] - right_bbox[1])
        x_overlap = max(
            0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0])
        )
        y_overlap = max(
            0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1])
        )
        x_gap = max(
            0, max(left_bbox[0], right_bbox[0]) - min(left_bbox[2], right_bbox[2])
        )
        y_gap = max(
            0, max(left_bbox[1], right_bbox[1]) - min(left_bbox[3], right_bbox[3])
        )
        same_line = (
            y_overlap >= 0.40 * min(left_height, right_height)
            and x_gap <= 2.0 * max(left_height, right_height)
        )
        stacked_lines = (
            x_overlap >= 0.15 * min(left_width, right_width)
            and y_gap <= max(8.0, 1.35 * min(left_height, right_height))
        )
        return bool(same_line or stacked_lines)

    for index, left_id in enumerate(component_ids):
        left = evidence_by_component[left_id]
        left_evidence_id = str(left.get("evidence_id") or "")
        left_source = str(left.get("source") or "")
        left_bbox = tuple(int(value) for value in left.get("bbox_page") or ())
        for right_id in component_ids[index + 1 :]:
            right = evidence_by_component[right_id]
            right_evidence_id = str(right.get("evidence_id") or "")
            if left_evidence_id and left_evidence_id == right_evidence_id:
                if (
                    not (is_page_spanning(left) or is_page_spanning(right))
                    or semantic_neighbours(left, right)
                ):
                    union(left_id, right_id)
                continue
            if (
                left_source != "full_page_visual_container"
                or str(right.get("source") or "") != "full_page_visual_container"
            ):
                continue
            right_bbox = tuple(int(value) for value in right.get("bbox_page") or ())
            if len(left_bbox) != 4 or len(right_bbox) != 4:
                continue
            if (
                containment_fraction(left_bbox, right_bbox) >= 0.80
                and (
                    not (is_page_spanning(left) or is_page_spanning(right))
                    or semantic_neighbours(left, right)
                )
            ):
                union(left_id, right_id)

    members_by_root: dict[str, list[str]] = {}
    for component_id in component_ids:
        members_by_root.setdefault(find(component_id), []).append(component_id)

    roots_by_evidence_id: dict[str, set[str]] = {}
    for component_id in component_ids:
        evidence_id = str(
            evidence_by_component[component_id].get("evidence_id") or ""
        )
        if evidence_id:
            roots_by_evidence_id.setdefault(evidence_id, set()).add(
                find(component_id)
            )

    result: dict[str, str] = {}
    for members in members_by_root.values():
        evidence_ids = {
            str(evidence_by_component[component_id].get("evidence_id") or "")
            for component_id in members
        }
        if (
            len(evidence_ids) == 1
            and len(roots_by_evidence_id.get(next(iter(evidence_ids)), ())) == 1
        ):
            canonical_id = next(iter(evidence_ids))
        else:
            payload = [
                {
                    "component_id": component_id,
                    "evidence_id": evidence_by_component[component_id].get("evidence_id"),
                    "bbox_page": list(evidence_by_component[component_id].get("bbox_page") or ()),
                }
                for component_id in sorted(members)
            ]
            digest = sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
            canonical_id = f"full_page_visual_container_group:{digest}"
        for component_id in members:
            result[component_id] = canonical_id
    return result


def recover_full_page_visual_container(
    image_rgb: np.ndarray,
    *,
    owner_id: str,
    semantic_body_bbox_page: BBox,
    source_replacement_bbox_page: BBox,
) -> dict[str, Any] | None:
    """Recover an enclosing visual region without promoting cleanup geometry.

    Capacity comes from the semantic component envelope.  A pre-inpaint edge
    crossing that envelope authenticates it as visual layout evidence; the
    cleanup footprint is validated but never used to size the container.
    """

    if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("container recovery requires an RGB logical-page image")
    del owner_id
    height, width = image_rgb.shape[:2]
    semantic = _bbox(semantic_body_bbox_page, width=width, height=height)
    cleanup = _bbox(source_replacement_bbox_page, width=width, height=height)
    detected = _threshold_visual_container(image_rgb, semantic_bbox=semantic)
    if detected is not None:
        support_fraction = float(detected["support_fraction"])
        fill_ratio = float(detected["fill_ratio"])
        return _visual_container_evidence(
            image_rgb,
            bbox_page=tuple(detected["bbox_page"]),
            confidence=min(0.98, 0.72 + support_fraction * 0.16 + fill_ratio * 0.10),
            threshold=int(detected["threshold"]),
            polarity=str(detected["polarity"]),
            component_area=int(detected["component_area"]),
        )
    # Capacity is derived from semantic component geometry, never from the
    # cleanup footprint.  Pixels only authenticate that the envelope reaches a
    # real visual boundary instead of being a blind geometric expansion.
    recovered = _proportional_fallback(semantic, width=width, height=height)
    if recovered == semantic or recovered == (0, 0, width, height):
        return None
    x1, y1, x2, y2 = recovered
    image_bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    border_width = max(3, int(min(edges.shape[:2]) * 0.06))
    border = np.zeros_like(edges, dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    edge_density = float(np.count_nonzero(edges[border])) / float(
        max(1, np.count_nonzero(border))
    )
    if edge_density < 0.008:
        return None
    return _visual_container_evidence(
        image_rgb,
        bbox_page=recovered,
        confidence=min(0.95, 0.70 + edge_density * 4.0),
        edge_density=round(edge_density, 6),
    )


def recover_component_visual_container(
    image_rgb: np.ndarray,
    *,
    component_id: str,
    glyph_bbox_page: BBox,
    glyph_polygon_page,
) -> dict[str, Any]:
    """Recover a visual container or derive an executable support-local one."""

    del glyph_polygon_page
    primary = recover_full_page_visual_container(
        image_rgb,
        owner_id=component_id,
        semantic_body_bbox_page=glyph_bbox_page,
        source_replacement_bbox_page=glyph_bbox_page,
    )
    if primary is not None:
        return {
            **primary,
            "semantic_bbox_page": tuple(int(value) for value in glyph_bbox_page),
        }
    height, width = image_rgb.shape[:2]
    conservative = _proportional_fallback(
        _bbox(glyph_bbox_page, width=width, height=height),
        width=width,
        height=height,
    )
    payload = {
        "component_id": component_id,
        "source": "conservative_support_local_container",
        "bbox_page": list(conservative),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "evidence_id": f"{component_id}:support_local_container:{digest}",
        "source": "conservative_support_local_container",
        "bbox_page": conservative,
        "semantic_bbox_page": tuple(int(value) for value in glyph_bbox_page),
        "page_shape": tuple(int(value) for value in image_rgb.shape),
        "confidence": 0.60,
        "conservative": True,
    }
