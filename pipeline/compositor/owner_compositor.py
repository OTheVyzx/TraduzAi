"""Deterministic, fail-closed composition of owner-scoped page mutations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
from typing import Any

import cv2
import numpy as np

try:
    from ownership.model import (
        OwnerCompositionConflict,
        OwnerGlyphPatch,
        OwnerMutation,
        PageCompositionResult,
    )
except ImportError:  # pragma: no cover - supports package imports
    from ..ownership.model import (
        OwnerCompositionConflict,
        OwnerGlyphPatch,
        OwnerMutation,
        PageCompositionResult,
    )


class OwnerCompositionError(ValueError):
    """Raised when input evidence is invalid before page composition."""

    def __init__(
        self,
        message: str,
        *,
        conflicts: Sequence[OwnerCompositionConflict] = (),
    ) -> None:
        self.conflicts = tuple(conflicts)
        super().__init__(message)


@dataclass(frozen=True)
class _ValidatedMutation:
    value: OwnerMutation
    owner_id: str
    page_id: str
    result_rgb: np.ndarray
    action_mask: np.ndarray
    changed: np.ndarray
    protected_art_mask: np.ndarray
    fingerprint: str


@dataclass(frozen=True)
class _ValidatedGlyphPatch:
    value: OwnerGlyphPatch
    owner_id: str
    page_id: str
    result_rgb: np.ndarray
    glyph_mask: np.ndarray
    changed: np.ndarray
    fingerprint: str


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_identity(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OwnerCompositionError(f"{label} must be a canonical non-empty string")
    return value


def _canonical_hash(value: object, *, label: str) -> str:
    digest = _canonical_identity(value, label=label)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise OwnerCompositionError(f"{label} must be a canonical sha256")
    return digest


def _action_mask_ref_matches_owner(owner_id: str, action_mask_ref: str) -> bool:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner_id).strip("._")
    if not safe or "\\" in action_mask_ref:
        return False
    identity_hash = sha256(owner_id.encode("utf-8")).hexdigest()[:12]
    expected_segment = f"{safe}--{identity_hash}"
    parts = PurePosixPath(action_mask_ref).parts
    return bool(
        len(parts) == 4
        and parts[0] == "owner_masks"
        and parts[1] == expected_segment
        and parts[2] not in {"", ".", ".."}
        and parts[3] == "action_mask.png"
    )


def _canonical_rgb(
    value: object,
    *,
    label: str,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[2] != 3
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or (shape is not None and tuple(value.shape) != tuple(shape))
    ):
        raise OwnerCompositionError(f"{label} must be a canonical RGB uint8 page")
    return np.ascontiguousarray(value, dtype=np.uint8)


def _canonical_mask(
    value: object,
    *,
    label: str,
    shape: tuple[int, int],
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 2
        or tuple(value.shape) != tuple(shape)
        or not np.all((value == 0) | (value == 255))
    ):
        raise OwnerCompositionError(
            f"{label} must be a canonical binary uint8 page mask"
        )
    return np.ascontiguousarray(value, dtype=np.uint8)


def _canonical_bbox(
    value: object,
    *,
    label: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
    ):
        raise OwnerCompositionError(f"{label} must contain four canonical integers")
    x1, y1, x2, y2 = (int(item) for item in value)
    height, width = shape
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2 or x2 > width or y2 > height:
        raise OwnerCompositionError(f"{label} escapes canonical page geometry")
    return x1, y1, x2, y2


def _mask_bbox(mask: np.ndarray, *, label: str) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        raise OwnerCompositionError(f"{label} is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _canonical_counter(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OwnerCompositionError(f"{label} must be a canonical non-negative integer")
    return value


def _contains_bbox(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    return bool(
        outer[0] <= inner[0] < inner[2] <= outer[2]
        and outer[1] <= inner[1] < inner[3] <= outer[3]
    )


def _canonical_safe_polygon(
    value: object,
    *,
    shape: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise OwnerCompositionError(
            "glyph render safe polygon must contain at least three points"
        )
    height, width = shape
    points: list[tuple[int, int]] = []
    for raw_point in value:
        if (
            not isinstance(raw_point, (list, tuple))
            or len(raw_point) != 2
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in raw_point
            )
        ):
            raise OwnerCompositionError(
                "glyph render safe polygon contains a non-canonical point"
            )
        x, y = int(raw_point[0]), int(raw_point[1])
        if x < 0 or y < 0 or x >= width or y >= height:
            raise OwnerCompositionError("glyph render safe polygon escapes page geometry")
        points.append((x, y))
    contour = np.asarray(points, dtype=np.int32)
    if float(abs(cv2.contourArea(contour))) <= 0.0:
        raise OwnerCompositionError("glyph render safe polygon has no area")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return tuple(points), mask


def _polygon_sha256(points: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps(points, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint(kind: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = sha256()
    digest.update(b"traduzai.owner_compositor.fingerprint.v1\0")
    digest.update(kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(encoded)
    return digest.hexdigest()


def _validate_mutation(
    mutation: object,
    *,
    original: np.ndarray,
    original_sha256: str,
    global_protected: np.ndarray,
) -> _ValidatedMutation:
    if not isinstance(mutation, OwnerMutation):
        raise OwnerCompositionError("mutation must be an OwnerMutation")
    owner_id = _canonical_identity(mutation.owner_id, label="mutation owner_id")
    page_id = _canonical_identity(mutation.page_id, label="mutation page_id")
    if mutation.coordinate_space != "page":
        raise OwnerCompositionError("mutation coordinate space must be page")
    if mutation.color_space != "RGB":
        raise OwnerCompositionError("mutation color space must be RGB")
    if mutation.projection_role != "executor":
        raise OwnerCompositionError("context_only mutation cannot write; executor required")
    execution_tile_id = _canonical_identity(
        mutation.execution_tile_id,
        label="mutation execution_tile_id",
    )
    action_mask_ref = _canonical_identity(
        mutation.action_mask_ref,
        label="mutation action_mask_ref",
    )
    if not _action_mask_ref_matches_owner(owner_id, action_mask_ref):
        raise OwnerCompositionError("mutation action_mask_ref is not bound to owner_id")
    _canonical_identity(mutation.engine, label="mutation engine")
    component_hash = _canonical_hash(
        mutation.component_geometry_sha256,
        label="mutation component_geometry_sha256",
    )

    result = _canonical_rgb(
        mutation.result_rgb,
        label="mutation result_rgb",
        shape=tuple(original.shape),
    )
    page_shape = tuple(original.shape[:2])
    action_mask = _canonical_mask(
        mutation.action_mask,
        label="mutation action_mask",
        shape=page_shape,
    )
    changed_mask = _canonical_mask(
        mutation.changed_mask,
        label="mutation changed_mask",
        shape=page_shape,
    )
    local_protected = _canonical_mask(
        mutation.protected_art_mask,
        label="mutation protected_art_mask",
        shape=page_shape,
    )
    action_bbox = _mask_bbox(action_mask, label="mutation action_mask")
    owner_bbox = _canonical_bbox(
        mutation.owner_bbox_page,
        label="mutation owner_bbox_page",
        shape=page_shape,
    )
    crop_bbox = _canonical_bbox(
        mutation.engine_crop_bbox_page,
        label="mutation engine_crop_bbox_page",
        shape=page_shape,
    )
    if not _contains_bbox(owner_bbox, action_bbox):
        raise OwnerCompositionError("mutation action mask escapes owner bbox")
    if not _contains_bbox(crop_bbox, action_bbox):
        raise OwnerCompositionError("mutation action mask escapes engine crop")

    if mutation.before_sha256 != original_sha256:
        raise OwnerCompositionError("mutation before hash does not match original")
    if mutation.after_sha256 != _array_sha256(result):
        raise OwnerCompositionError("mutation result hash mismatch")
    if mutation.action_mask_sha256 != _array_sha256(action_mask):
        raise OwnerCompositionError("mutation action mask hash mismatch")
    if mutation.changed_mask_sha256 != _array_sha256(changed_mask):
        raise OwnerCompositionError("mutation changed mask hash mismatch")
    protected_hash = _canonical_hash(
        mutation.protected_art_mask_sha256,
        label="mutation protected_art_mask_sha256",
    )
    if protected_hash != _array_sha256(local_protected):
        raise OwnerCompositionError("mutation protected art mask hash mismatch")

    actual_changed = np.any(result != original, axis=2)
    if not np.any(actual_changed):
        raise OwnerCompositionError("mutation changed no pixels")
    if not np.array_equal(actual_changed, changed_mask > 0):
        raise OwnerCompositionError("mutation changed mask does not match pixel delta")
    outside_action = actual_changed & (action_mask == 0)
    if np.any(outside_action):
        raise OwnerCompositionError("mutation changed pixels outside owner action mask")
    if np.any((action_mask > 0) & (local_protected > 0)):
        raise OwnerCompositionError("mutation action mask touches protected art")
    if np.any((action_mask > 0) & (global_protected > 0)):
        raise OwnerCompositionError("mutation action mask touches global protected art")
    protected_changed = actual_changed & (local_protected > 0)

    counters = {
        "mask_pixels": _canonical_counter(
            mutation.mask_pixels,
            label="mutation mask_pixels",
        ),
        "changed_pixels": _canonical_counter(
            mutation.changed_pixels,
            label="mutation changed_pixels",
        ),
        "changed_outside_owner_pixels": _canonical_counter(
            mutation.changed_outside_owner_pixels,
            label="mutation changed_outside_owner_pixels",
        ),
        "protected_art_changed_pixels": _canonical_counter(
            mutation.protected_art_changed_pixels,
            label="mutation protected_art_changed_pixels",
        ),
    }
    expected_counters = {
        "mask_pixels": int(np.count_nonzero(action_mask)),
        "changed_pixels": int(np.count_nonzero(actual_changed)),
        "changed_outside_owner_pixels": int(np.count_nonzero(outside_action)),
        "protected_art_changed_pixels": int(np.count_nonzero(protected_changed)),
    }
    if counters != expected_counters:
        raise OwnerCompositionError("mutation pixel counters do not match evidence")

    fingerprint = _fingerprint(
        "cleanup",
        {
            "owner_id": owner_id,
            "page_id": page_id,
            "coordinate_space": mutation.coordinate_space,
            "color_space": mutation.color_space,
            "projection_role": mutation.projection_role,
            "execution_tile_id": execution_tile_id,
            "action_mask_ref": action_mask_ref,
            "engine": mutation.engine,
            "engine_crop_bbox_page": crop_bbox,
            "owner_bbox_page": owner_bbox,
            "component_geometry_sha256": component_hash,
            "before_sha256": mutation.before_sha256,
            "after_sha256": mutation.after_sha256,
            "action_mask_sha256": mutation.action_mask_sha256,
            "changed_mask_sha256": mutation.changed_mask_sha256,
            "protected_art_mask_sha256": protected_hash,
            "counters": counters,
            "residual_score": mutation.residual_score,
            "changed_mask_ref": mutation.changed_mask_ref,
        },
    )
    return _ValidatedMutation(
        value=mutation,
        owner_id=owner_id,
        page_id=page_id,
        result_rgb=result,
        action_mask=action_mask,
        changed=actual_changed,
        protected_art_mask=local_protected,
        fingerprint=fingerprint,
    )


def _validate_glyph_patch(
    glyph_patch: object,
    *,
    original: np.ndarray,
    original_sha256: str,
    mutation: _ValidatedMutation | None,
    global_protected: np.ndarray,
) -> _ValidatedGlyphPatch:
    if not isinstance(glyph_patch, OwnerGlyphPatch):
        raise OwnerCompositionError("glyph patch must be an OwnerGlyphPatch")
    owner_id = _canonical_identity(glyph_patch.owner_id, label="glyph owner_id")
    page_id = _canonical_identity(glyph_patch.page_id, label="glyph page_id")
    if glyph_patch.coordinate_space != "page":
        raise OwnerCompositionError("glyph coordinate space must be page")
    if glyph_patch.color_space != "RGB":
        raise OwnerCompositionError("glyph color space must be RGB")
    if glyph_patch.projection_role != "executor":
        raise OwnerCompositionError("context_only glyph cannot write; executor required")
    execution_tile_id = _canonical_identity(
        glyph_patch.execution_tile_id,
        label="glyph execution_tile_id",
    )
    if glyph_patch.render_completed is not True or glyph_patch.fit_status != "ok":
        raise OwnerCompositionError("glyph render is not complete and fit-safe")

    baseline = original if mutation is None else mutation.result_rgb
    baseline_sha256 = original_sha256 if mutation is None else mutation.value.after_sha256
    if mutation is not None:
        if mutation.owner_id != owner_id or mutation.page_id != page_id:
            raise OwnerCompositionError("glyph mutation chain identity mismatch")
        if mutation.value.execution_tile_id != execution_tile_id:
            raise OwnerCompositionError("glyph mutation chain execution tile mismatch")
        if (
            glyph_patch.component_geometry_sha256
            != mutation.value.component_geometry_sha256
        ):
            raise OwnerCompositionError("glyph component geometry revision mismatch")
    component_hash = _canonical_hash(
        glyph_patch.component_geometry_sha256,
        label="glyph component_geometry_sha256",
    )

    result = _canonical_rgb(
        glyph_patch.result_rgb,
        label="glyph result_rgb",
        shape=tuple(original.shape),
    )
    page_shape = tuple(original.shape[:2])
    glyph_mask = _canonical_mask(
        glyph_patch.glyph_mask,
        label="glyph mask",
        shape=page_shape,
    )
    glyph_bbox = _canonical_bbox(
        glyph_patch.glyph_bbox_page,
        label="glyph bbox_page",
        shape=page_shape,
    )
    if glyph_bbox != _mask_bbox(glyph_mask, label="glyph mask"):
        raise OwnerCompositionError("glyph bbox does not match glyph mask")
    safe_polygon, safe_polygon_mask = _canonical_safe_polygon(
        glyph_patch.render_safe_polygon_page,
        shape=page_shape,
    )
    polygon_hash = _canonical_hash(
        glyph_patch.render_safe_polygon_sha256,
        label="glyph render_safe_polygon_sha256",
    )
    if polygon_hash != _polygon_sha256(safe_polygon):
        raise OwnerCompositionError("glyph render safe polygon hash mismatch")
    if np.any((glyph_mask > 0) & (safe_polygon_mask == 0)):
        raise OwnerCompositionError("glyph mask escapes render safe polygon")

    if glyph_patch.before_sha256 != baseline_sha256:
        raise OwnerCompositionError("glyph before hash does not match owner baseline")
    if glyph_patch.after_sha256 != _array_sha256(result):
        raise OwnerCompositionError("glyph result hash mismatch")
    if glyph_patch.glyph_mask_sha256 != _array_sha256(glyph_mask):
        raise OwnerCompositionError("glyph mask hash mismatch")
    actual_changed = np.any(result != baseline, axis=2)
    if not np.any(actual_changed):
        raise OwnerCompositionError("glyph render changed no pixels")
    if not np.array_equal(actual_changed, glyph_mask > 0):
        raise OwnerCompositionError("glyph mask does not match render pixel delta")
    changed_outside = actual_changed & (glyph_mask == 0)
    outside_count = _canonical_counter(
        glyph_patch.changed_outside_glyph_mask_pixels,
        label="glyph changed_outside_glyph_mask_pixels",
    )
    if outside_count != int(np.count_nonzero(changed_outside)):
        raise OwnerCompositionError("glyph outside-mask counter mismatch")
    if np.any((glyph_mask > 0) & (global_protected > 0)):
        raise OwnerCompositionError("glyph mask touches global protected art")
    if mutation is not None and np.any(
        (glyph_mask > 0) & (mutation.protected_art_mask > 0)
    ):
        raise OwnerCompositionError("glyph mask touches owner protected art")

    fingerprint = _fingerprint(
        "glyph",
        {
            "owner_id": owner_id,
            "page_id": page_id,
            "coordinate_space": glyph_patch.coordinate_space,
            "color_space": glyph_patch.color_space,
            "projection_role": glyph_patch.projection_role,
            "execution_tile_id": execution_tile_id,
            "render_completed": glyph_patch.render_completed,
            "fit_status": glyph_patch.fit_status,
            "before_sha256": glyph_patch.before_sha256,
            "after_sha256": glyph_patch.after_sha256,
            "glyph_mask_sha256": glyph_patch.glyph_mask_sha256,
            "glyph_bbox_page": glyph_bbox,
            "render_safe_polygon_page": safe_polygon,
            "render_safe_polygon_sha256": polygon_hash,
            "component_geometry_sha256": component_hash,
            "changed_outside_glyph_mask_pixels": outside_count,
        },
    )
    return _ValidatedGlyphPatch(
        value=glyph_patch,
        owner_id=owner_id,
        page_id=page_id,
        result_rgb=result,
        glyph_mask=glyph_mask,
        changed=actual_changed,
        fingerprint=fingerprint,
    )


def _dedupe_mutations(
    mutations: Sequence[_ValidatedMutation],
) -> tuple[list[_ValidatedMutation], list[OwnerCompositionConflict]]:
    by_owner: dict[str, list[_ValidatedMutation]] = {}
    for mutation in mutations:
        by_owner.setdefault(mutation.owner_id, []).append(mutation)
    unique: list[_ValidatedMutation] = []
    conflicts: list[OwnerCompositionConflict] = []
    for owner_id in sorted(by_owner):
        candidates = by_owner[owner_id]
        fingerprints = {candidate.fingerprint for candidate in candidates}
        if len(fingerprints) != 1:
            conflicts.append(
                OwnerCompositionConflict(
                    code="divergent_duplicate_mutation",
                    phase="preflight",
                    owner_ids=(owner_id,),
                    message="One owner supplied divergent cleanup mutations.",
                )
            )
        unique.append(min(candidates, key=lambda candidate: candidate.fingerprint))
    return unique, conflicts


def _dedupe_glyphs(
    glyphs: Sequence[_ValidatedGlyphPatch],
) -> tuple[list[_ValidatedGlyphPatch], list[OwnerCompositionConflict]]:
    by_owner: dict[str, list[_ValidatedGlyphPatch]] = {}
    for glyph in glyphs:
        by_owner.setdefault(glyph.owner_id, []).append(glyph)
    unique: list[_ValidatedGlyphPatch] = []
    conflicts: list[OwnerCompositionConflict] = []
    for owner_id in sorted(by_owner):
        candidates = by_owner[owner_id]
        fingerprints = {candidate.fingerprint for candidate in candidates}
        if len(fingerprints) != 1:
            conflicts.append(
                OwnerCompositionConflict(
                    code="divergent_duplicate_glyph_patch",
                    phase="preflight",
                    owner_ids=(owner_id,),
                    message="One owner supplied divergent glyph patches.",
                )
            )
        unique.append(min(candidates, key=lambda candidate: candidate.fingerprint))
    return unique, conflicts


def _pixel_conflicts(
    mutations: Sequence[_ValidatedMutation],
    glyphs: Sequence[_ValidatedGlyphPatch],
) -> list[OwnerCompositionConflict]:
    conflicts: list[OwnerCompositionConflict] = []
    for index, left in enumerate(mutations):
        for right in mutations[index + 1 :]:
            if left.owner_id == right.owner_id:
                continue
            overlap = (left.action_mask > 0) & (right.action_mask > 0)
            pixel_count = int(np.count_nonzero(overlap))
            if pixel_count:
                conflicts.append(
                    OwnerCompositionConflict(
                        code="owner_pixel_conflict",
                        phase="cleanup",
                        owner_ids=tuple(sorted((left.owner_id, right.owner_id))),
                        pixel_count=pixel_count,
                        message="Cleanup action masks overlap between owners.",
                    )
                )
    for index, left in enumerate(glyphs):
        for right in glyphs[index + 1 :]:
            if left.owner_id == right.owner_id:
                continue
            overlap = (left.glyph_mask > 0) & (right.glyph_mask > 0)
            pixel_count = int(np.count_nonzero(overlap))
            if pixel_count:
                conflicts.append(
                    OwnerCompositionConflict(
                        code="owner_pixel_conflict",
                        phase="glyph",
                        owner_ids=tuple(sorted((left.owner_id, right.owner_id))),
                        pixel_count=pixel_count,
                        message="Glyph masks overlap between owners.",
                    )
                )
    for mutation in mutations:
        for glyph in glyphs:
            if mutation.owner_id == glyph.owner_id:
                continue
            overlap = (mutation.action_mask > 0) & (glyph.glyph_mask > 0)
            pixel_count = int(np.count_nonzero(overlap))
            if pixel_count:
                conflicts.append(
                    OwnerCompositionConflict(
                        code="owner_pixel_conflict",
                        phase="cross_phase",
                        owner_ids=tuple(sorted((mutation.owner_id, glyph.owner_id))),
                        pixel_count=pixel_count,
                        message="Cleanup and glyph claims overlap between owners.",
                    )
                )
    return conflicts


def _empty_owner_map(
    shape: tuple[int, int],
    owner_ids: Sequence[str],
) -> np.ndarray:
    width = max((len(owner_id) for owner_id in owner_ids), default=1)
    return np.full(shape, "", dtype=f"<U{width}")


def _blocked_result(
    original: np.ndarray,
    *,
    page_id: str | None,
    owner_ids: Sequence[str],
    conflicts: Sequence[OwnerCompositionConflict],
) -> PageCompositionResult:
    ordered_conflicts = tuple(
        sorted(
            conflicts,
            key=lambda conflict: (
                conflict.phase,
                conflict.code,
                conflict.owner_ids,
                conflict.pixel_count,
            ),
        )
    )
    empty_cleanup = _empty_owner_map(tuple(original.shape[:2]), owner_ids)
    empty_glyph = _empty_owner_map(tuple(original.shape[:2]), owner_ids)
    return PageCompositionResult(
        final_rgb=original,
        cleanup_owner_map=empty_cleanup,
        glyph_owner_map=empty_glyph,
        conflicts=ordered_conflicts,
        write_counts={
            "cleanup_pixels": 0,
            "glyph_pixels": 0,
            "final_changed_pixels": 0,
            "owner_count": len(set(owner_ids)),
        },
        sha256=_array_sha256(original),
        page_id=page_id,
        committed=False,
    )


def compose_page(
    original_rgb: np.ndarray,
    owner_mutations: Sequence[OwnerMutation],
    owner_glyph_patches: Sequence[OwnerGlyphPatch],
    protected_art_mask: np.ndarray,
) -> PageCompositionResult:
    """Compose one page from validated owner deltas, independent of band order."""

    original = _canonical_rgb(original_rgb, label="original_rgb").copy()
    page_shape = tuple(original.shape[:2])
    global_protected = _canonical_mask(
        protected_art_mask,
        label="protected_art_mask",
        shape=page_shape,
    )
    try:
        mutation_inputs = tuple(owner_mutations)
        glyph_inputs = tuple(owner_glyph_patches)
    except TypeError as exc:
        raise OwnerCompositionError("owner artifacts must be finite sequences") from exc
    original_sha256 = _array_sha256(original)

    validated_mutations = [
        _validate_mutation(
            mutation,
            original=original,
            original_sha256=original_sha256,
            global_protected=global_protected,
        )
        for mutation in mutation_inputs
    ]
    unique_mutations, conflicts = _dedupe_mutations(validated_mutations)
    mutation_by_owner = {
        mutation.owner_id: mutation for mutation in unique_mutations
    }
    mutation_page_ids = {mutation.page_id for mutation in unique_mutations}
    if len(mutation_page_ids) > 1:
        raise OwnerCompositionError("owner artifacts contain mixed page ids")
    if conflicts:
        return _blocked_result(
            original,
            page_id=next(iter(mutation_page_ids), None),
            owner_ids=[mutation.owner_id for mutation in unique_mutations],
            conflicts=conflicts,
        )

    validated_glyphs = [
        _validate_glyph_patch(
            glyph,
            original=original,
            original_sha256=original_sha256,
            mutation=mutation_by_owner.get(
                str(getattr(glyph, "owner_id", "") or "")
            ),
            global_protected=global_protected,
        )
        for glyph in glyph_inputs
    ]
    unique_glyphs, glyph_conflicts = _dedupe_glyphs(validated_glyphs)
    conflicts.extend(glyph_conflicts)

    page_ids = {
        artifact.page_id
        for artifact in [*unique_mutations, *unique_glyphs]
    }
    if len(page_ids) > 1:
        raise OwnerCompositionError("owner artifacts contain mixed page ids")
    page_id = next(iter(page_ids), None)
    conflicts.extend(_pixel_conflicts(unique_mutations, unique_glyphs))
    owner_ids = [
        artifact.owner_id
        for artifact in [*unique_mutations, *unique_glyphs]
    ]
    if conflicts:
        return _blocked_result(
            original,
            page_id=page_id,
            owner_ids=owner_ids,
            conflicts=conflicts,
        )

    canvas = original.copy()
    cleanup_owner_map = _empty_owner_map(page_shape, owner_ids)
    glyph_owner_map = _empty_owner_map(page_shape, owner_ids)
    cleanup_pixels = 0
    glyph_pixels = 0

    for mutation in sorted(unique_mutations, key=lambda item: item.owner_id):
        canvas[mutation.changed] = mutation.result_rgb[mutation.changed]
        cleanup_owner_map[mutation.changed] = mutation.owner_id
        cleanup_pixels += int(np.count_nonzero(mutation.changed))

    for glyph in sorted(unique_glyphs, key=lambda item: item.owner_id):
        canvas[glyph.changed] = glyph.result_rgb[glyph.changed]
        glyph_owner_map[glyph.changed] = glyph.owner_id
        glyph_pixels += int(np.count_nonzero(glyph.changed))

    final_changed = np.any(canvas != original, axis=2)
    authorized = np.zeros(page_shape, dtype=bool)
    for mutation in unique_mutations:
        authorized |= mutation.action_mask > 0
    for glyph in unique_glyphs:
        authorized |= glyph.glyph_mask > 0
    if np.any(final_changed & ~authorized):
        raise OwnerCompositionError(
            "composed page changed pixels outside all owner masks"
        )

    return PageCompositionResult(
        final_rgb=canvas,
        cleanup_owner_map=cleanup_owner_map,
        glyph_owner_map=glyph_owner_map,
        conflicts=(),
        write_counts={
            "cleanup_pixels": cleanup_pixels,
            "glyph_pixels": glyph_pixels,
            "final_changed_pixels": int(np.count_nonzero(final_changed)),
            "owner_count": len(set(owner_ids)),
        },
        sha256=_array_sha256(canvas),
        page_id=page_id,
        committed=True,
    )
