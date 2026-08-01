"""Unified mask compositor for owner text and Latin SFX typography V2."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class GlyphRasterResult:
    status: str
    rgba: np.ndarray
    glyph_core_mask: np.ndarray
    effect_mask: np.ndarray
    glyph_core_envelope: tuple[int, int, int, int] | None
    effect_envelope: tuple[int, int, int, int] | None
    observed_attributes: dict[str, Any]
    abstained_attributes: dict[str, str]
    metrics: dict[str, Any]
    attribute_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    attribute_evidence_sha256: dict[str, str] = field(default_factory=dict)
    unavailable_attributes: dict[str, str] = field(default_factory=dict)

    @property
    def applied_attributes(self) -> dict[str, Any]:
        """Read-only compatibility alias for pre-R5 renderer consumers."""

        return self.observed_attributes


def _binary(mask: np.ndarray, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise ValueError("glyph masks must be two-dimensional")
    if shape is not None and raw.shape != shape:
        raise ValueError("safe mask must match glyph mask dimensions")
    return np.where(raw > 0, 255, 0).astype(np.uint8)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(np.where(mask > 0, 255, 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return (x, y, x + width, y + height)


def _finite(style: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(style.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _place_centered(canvas: np.ndarray, crop: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    output = np.zeros_like(canvas)
    x1 = int(round(center[0] - crop.shape[1] / 2.0))
    y1 = int(round(center[1] - crop.shape[0] / 2.0))
    source_x1, source_y1 = max(0, -x1), max(0, -y1)
    target_x1, target_y1 = max(0, x1), max(0, y1)
    width = min(crop.shape[1] - source_x1, output.shape[1] - target_x1)
    height = min(crop.shape[0] - source_y1, output.shape[0] - target_y1)
    if width > 0 and height > 0:
        output[target_y1 : target_y1 + height, target_x1 : target_x1 + width] = crop[
            source_y1 : source_y1 + height,
            source_x1 : source_x1 + width,
        ]
    return output


def _transform_core(
    glyph: np.ndarray,
    style: Mapping[str, Any],
    x_height: float,
) -> tuple[np.ndarray, dict[str, float]]:
    transformed = glyph.copy()
    bbox = _bbox(transformed)
    if bbox is None:
        raise ValueError("glyph mask has no core pixels")
    center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    tracking_xh = max(-0.25, min(1.0, _finite(style, "tracking_xh", 0.0)))
    if abs(tracking_xh) > 1e-6:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(transformed, 8)
        parts = [
            (label, float(centroids[label][0]))
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= 3
        ]
        parts.sort(key=lambda item: item[1])
        tracked = np.zeros_like(transformed)
        midpoint = (len(parts) - 1) / 2.0
        step = tracking_xh * x_height
        for index, (label, _centroid_x) in enumerate(parts):
            dx = int(round((index - midpoint) * step))
            matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float32)
            shifted = cv2.warpAffine(
                np.where(labels == label, 255, 0).astype(np.uint8),
                matrix,
                (transformed.shape[1], transformed.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            tracked = np.maximum(tracked, shifted)
        transformed = tracked

    width_scale = max(0.45, min(1.65, _finite(style, "width_scale", 1.0)))
    scale_y = max(0.45, min(1.65, _finite(style, "scale_y", 1.0)))
    bbox = _bbox(transformed)
    if bbox is None:
        raise ValueError("tracking removed every glyph pixel")
    crop = transformed[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop = cv2.resize(
        crop,
        (
            max(1, int(round(crop.shape[1] * width_scale))),
            max(1, int(round(crop.shape[0] * scale_y))),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    transformed = _place_centered(transformed, crop, center)

    slant = max(-0.75, min(0.75, _finite(style, "slant_tangent", 0.0)))
    if abs(slant) > 1e-6:
        matrix = np.asarray(
            [[1.0, slant, -slant * center[1]], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        transformed = cv2.warpAffine(
            transformed,
            matrix,
            (transformed.shape[1], transformed.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
    rotation = max(-60.0, min(60.0, _finite(style, "rotation_deg", 0.0)))
    if abs(rotation) > 1e-6:
        matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        transformed = cv2.warpAffine(
            transformed,
            matrix,
            (transformed.shape[1], transformed.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
    return _binary(transformed), {
        "tracking_xh": tracking_xh,
        "slant_tangent": slant,
        "width_scale": width_scale,
        "scale_y": scale_y,
        "rotation_deg": rotation,
    }


def _color(value: object, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    raw = str(value or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(character * 2 for character in raw)
    if len(raw) == 6:
        try:
            return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            pass
    return fallback


def _effect_width(effect: Mapping[str, Any], x_height: float, default_px: int = 0) -> int:
    if effect.get("width_xh") is not None:
        try:
            return max(0, int(round(float(effect["width_xh"]) * x_height)))
        except (TypeError, ValueError):
            pass
    try:
        return max(0, int(round(float(effect.get("width_px", default_px)))))
    except (TypeError, ValueError):
        return max(0, default_px)


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def _shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return cv2.warpAffine(
        mask,
        np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    )


def _outside(mask: np.ndarray, safe: np.ndarray) -> int:
    return int(np.count_nonzero((mask > 0) & (safe == 0)))


def _paint_solid(rgba: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: np.ndarray | int = 255) -> None:
    active = mask > 0
    if not np.any(active):
        return
    rgba[active, :3] = color
    if isinstance(alpha, np.ndarray):
        rgba[active, 3] = np.maximum(rgba[active, 3], alpha[active])
    else:
        rgba[active, 3] = np.maximum(rgba[active, 3], int(alpha))


def _contains_xheight_units(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).endswith("_xh") or _contains_xheight_units(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_xheight_units(item) for item in value)
    return False


def _layer_pixels(
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: np.ndarray | int = 255,
) -> np.ndarray:
    pixels = np.zeros((*mask.shape, 4), dtype=np.uint8)
    _paint_solid(pixels, mask, color, alpha)
    return pixels


def _array_evidence_sha256(mask: np.ndarray, pixels: np.ndarray) -> str:
    canonical_mask = np.ascontiguousarray(_binary(mask))
    canonical_pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(b"traduzai.glyph-raster-observation.v2\0")
    digest.update(str(canonical_mask.shape).encode("ascii"))
    digest.update(canonical_mask.tobytes(order="C"))
    digest.update(canonical_pixels.tobytes(order="C"))
    return digest.hexdigest()


def _observed_color(pixels: np.ndarray, mask: np.ndarray) -> str:
    active = (np.asarray(mask) > 0) & (np.asarray(pixels)[:, :, 3] > 0)
    if not np.any(active):
        raise ValueError("observable raster layer has no painted pixels")
    rgb = np.median(np.asarray(pixels)[active, :3], axis=0)
    channels = tuple(int(round(float(value))) for value in rgb)
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def _measured_expansion_px(core: np.ndarray, layer: np.ndarray) -> float:
    outside = (np.asarray(layer) > 0) & (np.asarray(core) == 0)
    if not np.any(outside):
        return 0.0
    distance = cv2.distanceTransform(
        np.where(core > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    return round(float(np.max(distance[outside])), 3)


def _gradient_endpoints(pixels: np.ndarray, core: np.ndarray) -> list[str]:
    ys = np.nonzero(core > 0)[0]
    if not len(ys):
        raise ValueError("gradient layer has no core pixels")
    colors = []
    for y in (int(ys.min()), int(ys.max())):
        row_mask = np.zeros_like(core)
        row_mask[y, :] = core[y, :]
        colors.append(_observed_color(pixels, row_mask))
    return colors


def _observe_backend_layers(
    *,
    rgba: np.ndarray,
    core: np.ndarray,
    layers: Mapping[str, Any],
    style: Mapping[str, Any],
    transforms: Mapping[str, Any],
    rendered_x_height_px: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    observed: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}
    evidence_hashes: dict[str, str] = {}
    unavailable: dict[str, str] = {}

    for name in ("slant_tangent", "width_scale", "scale_y", "rotation_deg"):
        if name not in style or name not in transforms:
            continue
        value = float(transforms[name])
        observed[name] = 0.0 if value == 0 else value
        transform_mask = core
        digest = _array_evidence_sha256(transform_mask, rgba)
        evidence[name] = {
            "evidence_kind": "transformed_core_mask",
            "evidence_sha256": digest,
            "metrics": {"rendered_x_height_px": rendered_x_height_px},
        }
        evidence_hashes[name] = digest

    def normalized_rows(layer_name: str) -> list[Mapping[str, Any]]:
        raw = layers.get(layer_name)
        if isinstance(raw, Mapping):
            return [raw]
        if isinstance(raw, (list, tuple)):
            return [row for row in raw if isinstance(row, Mapping)]
        return []

    fill_rows = normalized_rows("fill")
    gradient_rows = normalized_rows("gradient")
    if gradient_rows:
        row = gradient_rows[0]
        mask = _binary(np.asarray(row["mask"]), shape=core.shape)
        pixels = np.asarray(row["pixels"], dtype=np.uint8)
        observed["gradient"] = _gradient_endpoints(pixels, mask)
        digest = _array_evidence_sha256(mask, pixels)
        evidence["gradient"] = {
            "evidence_kind": "layer_pixels_and_mask",
            "evidence_sha256": digest,
            "metrics": {},
        }
        evidence_hashes["gradient"] = digest
    elif fill_rows:
        row = fill_rows[0]
        mask = _binary(np.asarray(row["mask"]), shape=core.shape)
        pixels = np.asarray(row["pixels"], dtype=np.uint8)
        observed["fill"] = _observed_color(pixels, mask)
        digest = _array_evidence_sha256(mask, pixels)
        evidence["fill"] = {
            "evidence_kind": "layer_pixels_and_mask",
            "evidence_sha256": digest,
            "metrics": {},
        }
        evidence_hashes["fill"] = digest
    elif "fill" in style or "gradient" in style:
        unavailable["gradient" if "gradient" in style else "fill"] = "missing_observable_fill_layer"

    stroke_rows = normalized_rows("stroke")
    requested_strokes = style.get("multistroke")
    stroke_name = "multistroke" if isinstance(requested_strokes, (list, tuple)) else "stroke"
    if stroke_rows:
        values = []
        layer_hashes = []
        for row in stroke_rows:
            mask = _binary(np.asarray(row["mask"]), shape=core.shape)
            pixels = np.asarray(row["pixels"], dtype=np.uint8)
            values.append(
                {
                    "color": _observed_color(pixels, mask),
                    "width_px": _measured_expansion_px(core, mask),
                }
            )
            layer_hashes.append(_array_evidence_sha256(mask, pixels))
        observed[stroke_name] = values if stroke_name == "multistroke" else values[0]
        digest = hashlib.sha256("".join(layer_hashes).encode("ascii")).hexdigest()
        evidence[stroke_name] = {
            "evidence_kind": "layer_pixels_and_mask",
            "evidence_sha256": digest,
            "metrics": {"layer_count": len(values)},
        }
        evidence_hashes[stroke_name] = digest
    elif "stroke" in style or "multistroke" in style:
        unavailable[stroke_name] = "missing_observable_effect_layer"

    for name in ("glow", "shadow"):
        rows = normalized_rows(name)
        if not rows:
            if name in style:
                unavailable[name] = "missing_observable_effect_layer"
            continue
        row = rows[0]
        mask = _binary(np.asarray(row["mask"]), shape=core.shape)
        pixels = np.asarray(row["pixels"], dtype=np.uint8)
        value: dict[str, Any] = {"color": _observed_color(pixels, mask)}
        if name == "glow":
            value["width_px"] = _measured_expansion_px(core, mask)
        else:
            core_bbox, layer_bbox = _bbox(core), _bbox(mask)
            if core_bbox is not None and layer_bbox is not None:
                value["offset"] = [
                    round((layer_bbox[0] + layer_bbox[2] - core_bbox[0] - core_bbox[2]) / 2),
                    round((layer_bbox[1] + layer_bbox[3] - core_bbox[1] - core_bbox[3]) / 2),
                ]
        observed[name] = value
        digest = _array_evidence_sha256(mask, pixels)
        evidence[name] = {
            "evidence_kind": "layer_pixels_and_mask",
            "evidence_sha256": digest,
            "metrics": {},
        }
        evidence_hashes[name] = digest
    return observed, evidence, evidence_hashes, unavailable


def rasterize_v2_glyph_layers(
    glyph_mask: np.ndarray,
    safe_mask: np.ndarray,
    style: Mapping[str, Any],
    *,
    source_x_height_px: float | None = None,
    rendered_x_height_px: float | None = None,
    backend: Callable[[np.ndarray, np.ndarray, Mapping[str, Any], float], Mapping[str, Any]] | None = None,
    enforce: bool = False,
) -> GlyphRasterResult:
    """Transform and compose glyph/effect masks without clipping requested ink."""

    glyph = _binary(glyph_mask)
    safe = _binary(safe_mask, shape=glyph.shape)
    raw_x_height = rendered_x_height_px if rendered_x_height_px is not None else source_x_height_px
    if raw_x_height is None:
        raise ValueError("rendered_x_height_px is required")
    x_height = max(1.0, float(raw_x_height))
    if enforce and _contains_xheight_units(style):
        raise ValueError("rasterizer received unresolved execution units")
    if backend is not None:
        raw_output = backend(glyph.copy(), safe.copy(), style, x_height)
        if not isinstance(raw_output, Mapping):
            raise ValueError("glyph raster backend must return an observable mapping")
        rgba = np.asarray(raw_output.get("rgba"), dtype=np.uint8)
        core = _binary(np.asarray(raw_output.get("glyph_core_mask")), shape=glyph.shape)
        if rgba.shape != (*glyph.shape, 4):
            raise ValueError("glyph raster backend RGBA shape is invalid")
        layers = raw_output.get("layers")
        layers = layers if isinstance(layers, Mapping) else {}
        transforms = raw_output.get("transforms")
        transforms = transforms if isinstance(transforms, Mapping) else {}
        observed, evidence, evidence_hashes, unavailable = _observe_backend_layers(
            rgba=rgba,
            core=core,
            layers=layers,
            style=style,
            transforms=transforms,
            rendered_x_height_px=x_height,
        )
        outside = _outside(rgba[:, :, 3], safe)
        if outside:
            raise ValueError("glyph raster backend painted outside safe mask")
        effect = np.where((rgba[:, :, 3] > 0) & (core == 0), 255, 0).astype(np.uint8)
        return GlyphRasterResult(
            status="review_required" if unavailable else "applied",
            rgba=rgba,
            glyph_core_mask=core,
            effect_mask=effect,
            glyph_core_envelope=_bbox(core),
            effect_envelope=_bbox(effect),
            observed_attributes=observed,
            abstained_attributes={},
            metrics={
                "core_pixels_outside_safe": _outside(core, safe),
                "effect_pixels_outside_safe": _outside(effect, safe),
                "stroke_width_px": (
                    observed.get("stroke", {}).get("width_px", 0)
                    if isinstance(observed.get("stroke"), Mapping)
                    else 0
                ),
                "source_x_height_px": x_height,
            },
            attribute_evidence=evidence,
            attribute_evidence_sha256=evidence_hashes,
            unavailable_attributes=unavailable,
        )
    core, transform = _transform_core(glyph, style, x_height)
    applied: dict[str, Any] = {
        key: value
        for key, value in transform.items()
        if key in style
        and key in {"slant_tangent", "width_scale", "scale_y", "rotation_deg"}
    }
    abstained: dict[str, str] = {}
    core_outside = _outside(core, safe)
    empty_rgba = np.zeros((*core.shape, 4), dtype=np.uint8)
    if core_outside:
        return GlyphRasterResult(
            "review_required",
            empty_rgba,
            core,
            np.zeros_like(core),
            _bbox(core),
            None,
            applied,
            {"core": "glyph_core_envelope_outside_safe"},
            {
                "core_pixels_outside_safe": core_outside,
                "effect_pixels_outside_safe": 0,
                "stroke_width_px": 0,
                "source_x_height_px": x_height,
                "diagnostic_tracking_xh": transform["tracking_xh"],
            },
        )

    rgba = np.zeros((*core.shape, 4), dtype=np.uint8)
    accepted_effects: list[tuple[str, np.ndarray, tuple[int, int, int], np.ndarray | int]] = []
    effect_union = np.zeros_like(core)
    maximum_stroke_width = 0

    raw_multistroke = style.get("multistroke")
    strokes: list[Mapping[str, Any]] = []
    if isinstance(raw_multistroke, (list, tuple)):
        strokes = [item for item in raw_multistroke if isinstance(item, Mapping)]
    elif isinstance(style.get("stroke"), Mapping):
        strokes = [style["stroke"]]
    accepted_strokes: list[tuple[Mapping[str, Any], np.ndarray, int]] = []
    for stroke in strokes:
        width = _effect_width(stroke, x_height)
        candidate = _dilate(core, width)
        if _outside(candidate, safe):
            abstained["multistroke" if len(strokes) > 1 else "stroke"] = "effect_envelope_outside_safe"
            accepted_strokes = []
            break
        accepted_strokes.append((stroke, candidate, width))
        maximum_stroke_width = max(maximum_stroke_width, width)
    for stroke, candidate, _width in sorted(accepted_strokes, key=lambda item: -item[2]):
        accepted_effects.append(("stroke", candidate, _color(stroke.get("color"), (0, 0, 0)), 255))
        effect_union = np.maximum(effect_union, candidate)
    if accepted_strokes:
        name = "multistroke" if len(strokes) > 1 else "stroke"
        applied[name] = copy.deepcopy(raw_multistroke if len(strokes) > 1 else strokes[0])

    shadow = style.get("shadow")
    if isinstance(shadow, Mapping):
        raw_offset_xh = shadow.get("offset_xh")
        if isinstance(raw_offset_xh, (list, tuple)) and len(raw_offset_xh) >= 2:
            dx, dy = (int(round(float(raw_offset_xh[index]) * x_height)) for index in (0, 1))
        else:
            raw_offset = shadow.get("offset") or (2, 2)
            dx, dy = (int(round(float(raw_offset[index]))) for index in (0, 1))
        candidate = _shift(core, dx, dy)
        if _outside(candidate, safe):
            abstained["shadow"] = "effect_envelope_outside_safe"
        else:
            accepted_effects.insert(0, ("shadow", candidate, _color(shadow.get("color"), (0, 0, 0)), 210))
            effect_union = np.maximum(effect_union, candidate)
            applied["shadow"] = copy.deepcopy(dict(shadow))

    glow = style.get("glow")
    if isinstance(glow, Mapping):
        width = _effect_width(glow, x_height, 2)
        expanded = _dilate(core, max(1, width))
        blurred = cv2.GaussianBlur(expanded, (0, 0), sigmaX=max(1.0, width * 0.75))
        candidate = expanded
        if _outside(candidate, safe):
            abstained["glow"] = "effect_envelope_outside_safe"
        else:
            alpha = np.clip(blurred.astype(np.float32) * 0.78, 0, 220).astype(np.uint8)
            accepted_effects.insert(0, ("glow", candidate, _color(glow.get("color"), (255, 255, 255)), alpha))
            effect_union = np.maximum(effect_union, candidate)
            applied["glow"] = copy.deepcopy(dict(glow))

    for _name, mask, color, alpha in accepted_effects:
        _paint_solid(rgba, mask, color, alpha)

    gradient = style.get("gradient")
    if isinstance(gradient, (list, tuple)) and len(gradient) >= 2:
        top = np.asarray(_color(gradient[0], (0, 0, 0)), dtype=np.float32)
        bottom = np.asarray(_color(gradient[1], (255, 255, 255)), dtype=np.float32)
        core_bbox = _bbox(core)
        assert core_bbox is not None
        for y in range(core_bbox[1], core_bbox[3]):
            ratio = (y - core_bbox[1]) / max(1.0, float(core_bbox[3] - core_bbox[1] - 1))
            row_color = tuple(int(round(value)) for value in (top * (1.0 - ratio) + bottom * ratio))
            _paint_solid(rgba, np.where((core > 0) & (np.indices(core.shape)[0] == y), 255, 0).astype(np.uint8), row_color)
        applied["gradient"] = list(gradient[:2])
    else:
        fill = style.get("fill") or style.get("fill_color") or "#000000"
        _paint_solid(rgba, core, _color(fill, (0, 0, 0)))
        applied["fill"] = fill

    output_outside = _outside(rgba[:, :, 3], safe)
    if output_outside:
        raise AssertionError("glyph rasterizer composed pixels outside safe mask")
    status = "fallback" if abstained else "applied"
    effect_only = effect_union.copy()
    effect_only[core > 0] = 0
    layers: dict[str, list[dict[str, np.ndarray]]] = {}
    if accepted_strokes:
        layers["stroke"] = [
            {
                "mask": candidate,
                "pixels": _layer_pixels(
                    candidate,
                    _color(stroke.get("color"), (0, 0, 0)),
                ),
            }
            for stroke, candidate, _width in accepted_strokes
        ]
    for name in ("shadow", "glow"):
        rows = [item for item in accepted_effects if item[0] == name]
        if rows:
            _layer_name, layer_mask, layer_color, layer_alpha = rows[0]
            layers[name] = [
                {
                    "mask": layer_mask,
                    "pixels": _layer_pixels(layer_mask, layer_color, layer_alpha),
                }
            ]
    core_pixels = np.zeros_like(rgba)
    core_pixels[core > 0] = rgba[core > 0]
    layers["gradient" if "gradient" in applied else "fill"] = [
        {"mask": core, "pixels": core_pixels}
    ]
    observed, evidence, evidence_hashes, unavailable = _observe_backend_layers(
        rgba=rgba,
        core=core,
        layers=layers,
        style=style,
        transforms=transform,
        rendered_x_height_px=x_height,
    )
    for name in abstained:
        unavailable.pop(name, None)
    if unavailable:
        status = "review_required"
    stroke_metric = observed.get("stroke")
    if isinstance(stroke_metric, Mapping):
        maximum_stroke_width = int(round(float(stroke_metric.get("width_px") or 0)))
    elif isinstance(observed.get("multistroke"), list):
        maximum_stroke_width = int(
            round(
                max(
                    (float(item.get("width_px") or 0) for item in observed["multistroke"]),
                    default=0,
                )
            )
        )
    return GlyphRasterResult(
        status,
        rgba,
        core,
        effect_only,
        _bbox(core),
        _bbox(effect_union),
        observed,
        abstained,
        {
            "core_pixels_outside_safe": 0,
            "effect_pixels_outside_safe": 0,
            "stroke_width_px": maximum_stroke_width,
            "source_x_height_px": x_height,
            "diagnostic_tracking_xh": transform["tracking_xh"],
        },
        evidence,
        evidence_hashes,
        unavailable,
    )
