from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

import numpy as np

from ownership.model import (
    OwnerStyleRasterContract,
    owner_style_raster_contract_sha256,
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
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def valid_owner_style_raster_contract(
    *,
    owner_id: str = "owner_p001_fixture",
    page_id: str = "page_001",
) -> OwnerStyleRasterContract:
    source = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    before = source.copy()
    patch = source.copy()
    patch[1:3, 1:3] = (240, 240, 240)
    after = patch.copy()
    glyph_mask = np.zeros((4, 4), dtype=np.uint8)
    glyph_mask[1:3, 1:3] = 255
    requested_attributes = {
        "fill": "#F0F0F0",
        "font_name": "ComicNeue-Bold.ttf",
    }
    applied_attributes = {"fill": "#F0F0F0"}
    abstained_attributes = {"font_name": "insufficient_margin"}
    raw = {
        "schema_version": 1,
        "page_id": page_id,
        "owner_id": owner_id,
        "visual_profile_sha256": _json_sha256(
            {"owner_id": owner_id, "status": "applied"}
        ),
        "profile_component_geometry_sha256": _json_sha256(
            {"component_ids": ["component_fixture"]}
        ),
        "execution_component_geometry_sha256": _json_sha256(
            {"bbox_page": [0, 0, 4, 4], "shape": [4, 4]}
        ),
        "source_artifact_sha256": _array_sha256(source),
        "source_glyph_mask_sha256": _array_sha256(glyph_mask),
        "status": "applied",
        "backend": "python_ft2font",
        "backend_version": "fixture-v1",
        "capabilities": ("fill", "font_name"),
        "requested_attributes": requested_attributes,
        "applied_attributes": applied_attributes,
        "abstained_attributes": abstained_attributes,
        "glyph_core_envelope": {
            "bbox_page": [1, 1, 3, 3],
            "mask_sha256": _array_sha256(glyph_mask),
            "pixel_count": 4,
        },
        "effect_envelope": {
            "bbox_page": [0, 0, 4, 4],
            "mask_sha256": _array_sha256(np.full((4, 4), 255, dtype=np.uint8)),
            "pixel_count": 16,
        },
        "render_metrics": {
            "core_pixel_count": 4,
            "effect_pixel_count": 0,
        },
        "segments": (),
        "rendered_before_sha256": _array_sha256(before),
        "rendered_patch_sha256": _array_sha256(patch),
        "rendered_after_sha256": _array_sha256(after),
    }
    raw["contract_sha256"] = owner_style_raster_contract_sha256(raw)
    return OwnerStyleRasterContract(**raw)
