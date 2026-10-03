"""Typed, verified snapshots of inputs consumed before owner execution.

This format never unpickles objects. It preserves masks and metadata arrays in
an NPZ file and binds both files to the Vision index by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from strip.types import BBox as StripBBox
from strip.types import Balloon, Band


_TAG = "__vision_stage_cache_type__"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pack(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot enter the Vision stage cache")
        key = f"array_{len(arrays)}"
        arrays[key] = value
        return {_TAG: "array", "key": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, StripBBox):
        return {_TAG: "bbox", "xyxy": [value.x1, value.y1, value.x2, value.y2]}
    if isinstance(value, Path):
        return {_TAG: "path", "value": str(value)}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_pack(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_pack(item, arrays) for item in value]
    if isinstance(value, dict):
        if _TAG in value or not all(isinstance(key, str) for key in value):
            raise TypeError("invalid Vision stage cache dictionary")
        return {key: _pack(item, arrays) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported Vision stage cache value: {type(value).__name__}")


def _unpack(value: Any, arrays: Any) -> Any:
    if isinstance(value, list):
        return [_unpack(item, arrays) for item in value]
    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag == "array":
            return np.asarray(arrays[value["key"]])
        if tag == "bbox":
            return StripBBox(*map(int, value["xyxy"]))
        if tag == "path":
            return Path(value["value"])
        if tag == "tuple":
            return tuple(_unpack(item, arrays) for item in value["items"])
        if tag is not None:
            raise ValueError("unknown Vision stage cache type")
        return {key: _unpack(item, arrays) for key, item in value.items()}
    return value


def _balloon_payload(balloon: Balloon, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "strip_bbox": _pack(balloon.strip_bbox, arrays),
        "confidence": float(balloon.confidence),
        "lobe_count": int(balloon.lobe_count),
        "metadata": _pack(balloon.metadata, arrays),
        "mask": _pack(balloon.mask, arrays),
    }


def _restore_balloon(payload: dict[str, Any], arrays: Any) -> Balloon:
    return Balloon(
        strip_bbox=_unpack(payload["strip_bbox"], arrays),
        confidence=float(payload["confidence"]),
        lobe_count=int(payload["lobe_count"]),
        metadata=_unpack(payload["metadata"], arrays),
        mask=_unpack(payload["mask"], arrays),
    )


def save_pre_ocr_stage(
    work_dir: Path,
    *,
    page_id: str,
    source_file_sha256: str,
    page_pixel_sha256: str,
    visual_config_sha256: str,
    balloons: list[Balloon],
    bands: list[Band],
    band_evidence_by_index: dict[int, Any],
) -> dict[str, Any]:
    root = work_dir / "vision" / "pre_ocr"
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "schema": "traduzai.vision-pre-ocr-stage.v1",
        "page_id": page_id,
        "source_file_sha256": source_file_sha256,
        "page_pixel_sha256": page_pixel_sha256,
        "visual_config_sha256": visual_config_sha256,
        "balloons": [_balloon_payload(item, arrays) for item in balloons],
        "bands": [
            {
                "y_top": int(band.y_top), "y_bottom": int(band.y_bottom),
                "balloons": [_balloon_payload(item, arrays) for item in band.balloons],
                "tile_id": band.tile_id,
                "strip_offset_xy": _pack(band.strip_offset_xy, arrays),
                "ocr_page": _pack(band_evidence_by_index[index].ocr_page, arrays),
                "perf": _pack(band_evidence_by_index[index].perf, arrays),
                "terminal_reason": band_evidence_by_index[index].terminal_reason,
            }
            for index, band in enumerate(bands)
        ],
    }
    stem = root / page_id
    temporary_arrays = root / f".{page_id}.{uuid4().hex}.npz"
    temporary_json = root / f".{page_id}.{uuid4().hex}.json"
    try:
        np.savez_compressed(temporary_arrays, **arrays)
        payload["arrays_sha256"] = _sha256(temporary_arrays)
        temporary_json.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary_arrays, stem.with_suffix(".npz"))
        os.replace(temporary_json, stem.with_suffix(".json"))
    finally:
        temporary_arrays.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    return {"json": stem.with_suffix(".json"), "npz": stem.with_suffix(".npz")}


def load_pre_ocr_stage(
    json_path: Path,
    npz_path: Path,
    *,
    page_id: str,
    source_file_sha256: str,
    page_pixel_sha256: str,
    visual_config_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "traduzai.vision-pre-ocr-stage.v1":
        raise ValueError("stage_cache_version_changed")
    if payload.get("page_id") != page_id or payload.get("source_file_sha256") != source_file_sha256 or payload.get("page_pixel_sha256") != page_pixel_sha256:
        raise ValueError("stage_cache_source_changed")
    if payload.get("visual_config_sha256") != visual_config_sha256:
        raise ValueError("stage_cache_visual_config_changed")
    if payload.get("arrays_sha256") != _sha256(npz_path):
        raise ValueError("stage_cache_arrays_changed")
    with np.load(npz_path, allow_pickle=False) as arrays:
        balloons = [_restore_balloon(item, arrays) for item in payload["balloons"]]
        bands = []
        for item in payload["bands"]:
            band = Band(
                y_top=int(item["y_top"]), y_bottom=int(item["y_bottom"]),
                balloons=[_restore_balloon(value, arrays) for value in item["balloons"]],
                tile_id=item["tile_id"],
                strip_offset_xy=_unpack(item["strip_offset_xy"], arrays),
            )
            bands.append({
                "band": band, "ocr_page": _unpack(item["ocr_page"], arrays),
                "perf": _unpack(item["perf"], arrays),
                "terminal_reason": item["terminal_reason"],
            })
    return {"balloons": balloons, "bands": bands}
