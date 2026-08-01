from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import numpy as np

from ownership.model import (
    OwnerStyleRasterContract,
    owner_style_raster_contract_sha256,
    owner_style_raster_segment_sha256,
)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _masked_pixels_sha256(image: np.ndarray, mask: np.ndarray) -> str:
    canonical_image = np.ascontiguousarray(image, dtype=np.uint8)
    canonical_mask = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if canonical_image.shape[:2] != canonical_mask.shape:
        raise ValueError("fixture image and glyph mask shapes differ")
    digest = sha256()
    digest.update(b"traduzai.masked-rgb.v1\0")
    digest.update(_array_sha256(canonical_mask).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_image[canonical_mask > 0].tobytes(order="C"))
    return digest.hexdigest()


def valid_owner_style_raster_contract(
    *,
    owner_id: str = "owner_p001_fixture",
    page_id: str = "page_001",
    before: np.ndarray | None = None,
    result: np.ndarray | None = None,
    glyph_mask: np.ndarray | None = None,
    component_geometry_sha256: str | None = None,
    visual_profile_sha256: str | None = None,
    profile_component_geometry_sha256: str | None = None,
    source_artifact_sha256: str | None = None,
    source_glyph_mask_sha256: str | None = None,
    style_decision: dict[str, Any] | None = None,
) -> OwnerStyleRasterContract:
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    canonical_before = np.ascontiguousarray(
        source.copy() if before is None else before,
        dtype=np.uint8,
    )
    canonical_result = np.ascontiguousarray(
        canonical_before.copy() if result is None else result,
        dtype=np.uint8,
    )
    if result is None:
        canonical_result[1:3, 1:3] = (240, 240, 240)
    canonical_mask = (
        np.zeros(canonical_before.shape[:2], dtype=np.uint8)
        if glyph_mask is None
        else np.where(np.asarray(glyph_mask) > 0, 255, 0).astype(np.uint8)
    )
    if glyph_mask is None:
        canonical_mask[1:3, 1:3] = 255
    if canonical_before.shape != canonical_result.shape:
        raise ValueError("fixture before and result shapes differ")
    if canonical_before.shape[:2] != canonical_mask.shape:
        raise ValueError("fixture result and glyph mask shapes differ")
    execution_geometry_sha256 = component_geometry_sha256 or _json_sha256(
        {"bbox_page": [0, 0, canonical_before.shape[1], canonical_before.shape[0]]}
    )
    decision_applied = dict((style_decision or {}).get("applied_attributes") or {})
    decision_abstained = dict(
        (style_decision or {}).get("abstained_attributes") or {}
    )
    requested_attributes = (
        {
            name: decision_applied.get(name, "unknown")
            for name in sorted(set(decision_applied) | set(decision_abstained))
        }
        if style_decision is not None
        else {
            "fill": "#F0F0F0",
            "font_name": "ComicNeue-Bold.ttf",
        }
    )
    applied_attributes: dict[str, Any] = {}
    abstained_attributes = {
        name: str(
            decision_abstained.get(name) or "fixture_has_no_renderer_evidence"
        )
        for name in requested_attributes
    }
    raw = {
        "schema_version": 1,
        "page_id": page_id,
        "owner_id": owner_id,
        "visual_profile_sha256": visual_profile_sha256 or _json_sha256(
            {"owner_id": owner_id, "status": "applied"}
        ),
        "profile_component_geometry_sha256": (
            profile_component_geometry_sha256
            or _json_sha256({"component_ids": ["component_fixture"]})
        ),
        "execution_component_geometry_sha256": execution_geometry_sha256,
        "source_artifact_sha256": (
            source_artifact_sha256 or _array_sha256(canonical_before)
        ),
        "source_glyph_mask_sha256": (
            source_glyph_mask_sha256 or _array_sha256(canonical_mask)
        ),
        "status": "fallback",
        "backend": "python_ft2font",
        "backend_version": "fixture-v1",
        "capabilities": ("fill", "font_name"),
        "requested_attributes": requested_attributes,
        "applied_attributes": applied_attributes,
        "abstained_attributes": abstained_attributes,
        "glyph_core_envelope": {
            "bbox_page": _mask_bbox(canonical_mask),
            "mask_sha256": _array_sha256(canonical_mask),
            "pixel_count": int(np.count_nonzero(canonical_mask)),
        },
        "effect_envelope": {
            "bbox_page": [],
            "mask_sha256": _array_sha256(np.zeros_like(canonical_mask)),
            "pixel_count": 0,
        },
        "render_metrics": {
            "core_pixel_count": int(np.count_nonzero(canonical_mask)),
            "effect_pixel_count": 0,
        },
        "segments": (),
        "rendered_before_sha256": _array_sha256(canonical_before),
        "rendered_patch_sha256": _masked_pixels_sha256(
            canonical_result,
            canonical_mask,
        ),
        "rendered_after_sha256": _array_sha256(canonical_result),
    }
    raw["contract_sha256"] = owner_style_raster_contract_sha256(raw)
    return OwnerStyleRasterContract(**raw)


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return []
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def valid_owner_style_raster_segment(
    *,
    owner_id: str = "owner_p001_fixture",
    visual_profile_sha256: str,
    segment_id: str = "region_0",
    order: int = 0,
    bbox_page: tuple[int, int, int, int] = (0, 0, 2, 2),
) -> dict[str, Any]:
    mask = np.zeros((4, 8), dtype=np.uint8)
    x1, y1, x2, y2 = bbox_page
    mask[y1:y2, x1:x2] = 255
    empty = np.zeros_like(mask)
    segment = {
        "segment_id": segment_id,
        "order": order,
        "owner_id": owner_id,
        "visual_profile_sha256": visual_profile_sha256,
        "bbox_page": list(bbox_page),
        "status": "fallback",
        "applied_attributes": {},
        "abstained_attributes": {
            "fill": "fixture_has_no_renderer_evidence"
        },
        "glyph_core_envelope": {
            "bbox_page": list(bbox_page),
            "mask_sha256": _array_sha256(mask),
            "pixel_count": int(np.count_nonzero(mask)),
        },
        "effect_envelope": {
            "bbox_page": [],
            "mask_sha256": _array_sha256(empty),
            "pixel_count": 0,
        },
        "rendered_before_sha256": _json_sha256(
            {"segment_id": segment_id, "phase": "before"}
        ),
        "rendered_patch_sha256": _json_sha256(
            {"segment_id": segment_id, "phase": "patch"}
        ),
        "rendered_after_sha256": _json_sha256(
            {"segment_id": segment_id, "phase": "after"}
        ),
    }
    segment["segment_sha256"] = owner_style_raster_segment_sha256(segment)
    return segment
