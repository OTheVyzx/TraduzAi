"""Hard raster safety against authorized writing area and protected artwork."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from ownership.hash_contract import canonical_json_sha256


RASTER_SAFETY_SCHEMA = "traduzai.raster-safety.v1"


def raster_mask_sha256(mask: np.ndarray) -> str:
    """Return the canonical hash used by persisted raster-safety masks."""

    normalized = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8) > 0, dtype=np.uint8)
    header = f"mask-v1:{normalized.shape[0]}x{normalized.shape[1]}:".encode("ascii")
    return sha256(header + normalized.tobytes()).hexdigest()


def _bbox(value: Sequence[Any]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("raster safety bbox must contain four coordinates")
    try:
        x1, y1, x2, y2 = (int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("raster safety bbox must contain integer coordinates") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError("raster safety bbox must have positive area")
    return x1, y1, x2, y2


def assess_raster_safety(
    *,
    alpha: np.ndarray,
    bbox: Sequence[Any],
    authorized_body_mask: np.ndarray,
    protected_art_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure every ink pixel against explicit page-space authority masks."""

    alpha_array = np.asarray(alpha)
    authorized = np.asarray(authorized_body_mask)
    protected = np.asarray(protected_art_mask)
    if alpha_array.ndim != 2:
        raise ValueError("raster safety alpha must be a two-dimensional mask")
    if authorized.ndim != 2 or protected.ndim != 2 or authorized.shape != protected.shape:
        raise ValueError("raster safety page masks must be two-dimensional and equal-sized")
    x1, y1, x2, y2 = _bbox(bbox)
    if x1 < 0 or y1 < 0 or x2 > authorized.shape[1] or y2 > authorized.shape[0]:
        raise ValueError("raster safety bbox lies outside page masks")
    if alpha_array.shape != (y2 - y1, x2 - x1):
        raise ValueError("raster safety alpha shape differs from bbox")

    ink = alpha_array > 0
    authorized_crop = authorized[y1:y2, x1:x2] > 0
    protected_crop = protected[y1:y2, x1:x2] > 0
    outside = int(np.count_nonzero(ink & ~authorized_crop))
    overlap = int(np.count_nonzero(ink & protected_crop))
    body = {
        "schema": RASTER_SAFETY_SCHEMA,
        "status": "pass" if outside == 0 and overlap == 0 else "review_required",
        "bbox": [x1, y1, x2, y2],
        "ink_pixel_count": int(np.count_nonzero(ink)),
        "alpha_sha256": raster_mask_sha256(ink),
        "authorized_body_sha256": raster_mask_sha256(authorized),
        "protected_art_sha256": raster_mask_sha256(protected),
        "outside_authorized_body_px": outside,
        "protected_art_overlap_px": overlap,
    }
    return body | {"evidence_sha256": canonical_json_sha256(body)}


def validate_raster_safety_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted safety receipt before a candidate claims PASS."""

    if not isinstance(value, Mapping):
        raise ValueError("candidate requires raster safety evidence")
    payload = dict(value)
    required = {
        "schema", "status", "bbox", "ink_pixel_count", "alpha_sha256",
        "authorized_body_sha256", "protected_art_sha256",
        "outside_authorized_body_px", "protected_art_overlap_px", "evidence_sha256",
    }
    if set(payload) != required or payload.get("schema") != RASTER_SAFETY_SCHEMA:
        raise ValueError("candidate requires complete raster safety evidence")
    for field in ("alpha_sha256", "authorized_body_sha256", "protected_art_sha256", "evidence_sha256"):
        digest = str(payload.get(field) or "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"raster safety {field} must be a lowercase SHA-256")
    _bbox(payload["bbox"])
    for field in ("ink_pixel_count", "outside_authorized_body_px", "protected_art_overlap_px"):
        if not isinstance(payload[field], int) or payload[field] < 0:
            raise ValueError(f"raster safety {field} must be a non-negative integer")
    body = {key: item for key, item in payload.items() if key != "evidence_sha256"}
    if canonical_json_sha256(body) != payload["evidence_sha256"]:
        raise ValueError("raster safety evidence hash mismatch")
    expected_status = (
        "pass"
        if payload["outside_authorized_body_px"] == 0
        and payload["protected_art_overlap_px"] == 0
        else "review_required"
    )
    if payload["status"] != expected_status:
        raise ValueError("raster safety status disagrees with collision counts")
    return payload
