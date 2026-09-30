"""Match source typography by rerendering frozen source text locally."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath


MATCH_SCORE_THRESHOLD = 0.68
MATCH_MARGIN_THRESHOLD = 0.025
NORMALIZED_CANVAS = (192, 512)
NORMALIZED_GLYPH_HEIGHT = 128


@dataclass(frozen=True)
class FontCatalogEntry:
    font_name: str
    path: Path
    roles: tuple[str, ...]
    license_status: str


@dataclass(frozen=True)
class FontMatchResult:
    value: str
    status: str
    confidence: float
    margin: float
    top_k: tuple[dict[str, Any], ...]
    abstention_reason: str
    cache_key: str
    catalog_version: str = ""
    normalization: Mapping[str, Any] = field(default_factory=dict)
    source_text_sha256: str = ""
    glyph_mask_sha256: str = ""

    @property
    def selected_font(self) -> str | None:
        return self.value if self.value != "unknown" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_font": self.selected_font,
            "value": self.value,
            "status": self.status,
            "confidence": self.confidence,
            "margin": self.margin,
            "top_k": [dict(item) for item in self.top_k],
            "abstention_reason": self.abstention_reason,
            "cache_key": self.cache_key,
            "catalog_version": self.catalog_version,
            "normalization": dict(self.normalization),
            "source_text_sha256": self.source_text_sha256,
            "glyph_mask_sha256": self.glyph_mask_sha256,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_font_catalog(fonts_dir: Path, font_map_path: Path) -> tuple[FontCatalogEntry, ...]:
    """Load only declared, existing local assets with explicit license status."""

    root = Path(fonts_dir).resolve()
    payload = json.loads(Path(font_map_path).read_text(encoding="utf-8"))
    entries: list[FontCatalogEntry] = []
    for raw in payload.get("available", []):
        if not isinstance(raw, Mapping) or not raw.get("detector", True):
            continue
        font_name = str(raw.get("arquivo") or "").strip()
        license_status = str(raw.get("license_status") or "").strip()
        roles = tuple(sorted({str(item).strip() for item in raw.get("roles", []) if str(item).strip()}))
        candidates = sorted(
            (item for item in root.rglob("*") if item.is_file() and item.name.casefold() == font_name.casefold()),
            key=lambda item: str(item).casefold(),
        )
        if not font_name or not candidates:
            raise ValueError(f"font catalog references missing file: {font_name!r}")
        if not license_status or not roles:
            raise ValueError(f"font catalog entry lacks license/role metadata: {font_name}")
        resolved = candidates[0].resolve()
        if root not in resolved.parents:
            raise ValueError(f"font catalog path escapes fonts root: {font_name}")
        entries.append(FontCatalogEntry(font_name, resolved, roles, license_status))
    names = [entry.font_name.casefold() for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("font catalog contains duplicate file names")
    return tuple(sorted(entries, key=lambda item: item.font_name.casefold()))


def _profile_transform(profile: Mapping[str, Any]) -> dict[str, float]:
    def finite(name: str, default: float) -> float:
        try:
            value = float(profile.get(name, default))
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    return {
        "rotation_deg": max(-45.0, min(45.0, finite("rotation_deg", 0.0))),
        "slant_tangent": max(-0.75, min(0.75, finite("slant_tangent", 0.0))),
        "width_scale": max(0.45, min(1.65, finite("width_scale", 1.0))),
        "scale_y": max(0.45, min(1.65, finite("scale_y", 1.0))),
    }


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("source font mask must be two-dimensional")
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 8:
        raise ValueError("source font mask has insufficient glyph pixels")
    x, y, width, height = cv2.boundingRect(points)
    crop = binary[y : y + height, x : x + width]
    scale = NORMALIZED_GLYPH_HEIGHT / max(1.0, float(height))
    resized_width = max(1, int(round(width * scale)))
    resized = cv2.resize(
        crop,
        (resized_width, NORMALIZED_GLYPH_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas_height, canvas_width = NORMALIZED_CANVAS
    if resized_width > canvas_width - 8:
        resized = cv2.resize(
            resized,
            (canvas_width - 8, NORMALIZED_GLYPH_HEIGHT),
            interpolation=cv2.INTER_NEAREST,
        )
    canvas = np.zeros(NORMALIZED_CANVAS, dtype=np.uint8)
    top = (canvas_height - resized.shape[0]) // 2
    left = (canvas_width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def render_source_text_mask(
    source_text: str,
    font_path: Path,
    *,
    profile: Mapping[str, Any],
) -> np.ndarray:
    """Render frozen source text and apply measured geometric transforms."""

    text = str(source_text or "")
    if not text.strip():
        raise ValueError("source_text must be non-empty")
    path = Path(font_path)
    if not path.is_file():
        raise ValueError(f"font file does not exist: {path}")
    text_path = TextPath(
        (0, 0),
        text,
        prop=FontProperties(fname=str(path), size=112),
        usetex=False,
    )
    bbox = text_path.get_extents()
    width = max(1, int(np.ceil(float(bbox.width))))
    height = max(1, int(np.ceil(float(bbox.height))))
    padding = 48
    mask = np.zeros((height + padding * 2, width + padding * 2), dtype=np.uint8)
    polygons: list[tuple[float, np.ndarray]] = []
    for polygon in text_path.to_polygons():
        if len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.float64)
        points[:, 0] = points[:, 0] - float(bbox.x0) + padding
        points[:, 1] = float(bbox.y1) - points[:, 1] + padding
        canonical = np.round(points).astype(np.int32)
        polygons.append((cv2.contourArea(canonical, oriented=True), canonical))
    if not polygons:
        raise ValueError("font produced no renderable source glyphs")
    exterior_sign = 1.0 if max(polygons, key=lambda item: abs(item[0]))[0] >= 0.0 else -1.0
    for signed_area, points in sorted(polygons, key=lambda item: -abs(item[0])):
        color = 255 if signed_area * exterior_sign >= 0.0 else 0
        cv2.fillPoly(mask, [points], color)
    transform = _profile_transform(profile)
    if abs(transform["slant_tangent"]) > 1e-6:
        extra = int(round(abs(transform["slant_tangent"]) * mask.shape[0])) + 4
        matrix = np.asarray(
            [[1.0, transform["slant_tangent"], extra if transform["slant_tangent"] < 0 else 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        mask = cv2.warpAffine(mask, matrix, (mask.shape[1] + extra, mask.shape[0]), flags=cv2.INTER_NEAREST)
    if abs(transform["width_scale"] - 1.0) > 1e-6:
        mask = cv2.resize(
            mask,
            (max(1, int(round(mask.shape[1] * transform["width_scale"]))), mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    if abs(transform["scale_y"] - 1.0) > 1e-6:
        mask = cv2.resize(
            mask,
            (mask.shape[1], max(1, int(round(mask.shape[0] * transform["scale_y"])))),
            interpolation=cv2.INTER_NEAREST,
        )
    if abs(transform["rotation_deg"]) > 1e-6:
        center = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
        matrix = cv2.getRotationMatrix2D(center, transform["rotation_deg"], 1.0)
        mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return _normalize_mask(mask)


def _catalog_contract(catalog: Sequence[FontCatalogEntry]) -> list[dict[str, Any]]:
    return [
        {
            "font_name": entry.font_name,
            "font_sha256": _file_sha256(entry.path),
            "license_status": entry.license_status,
            "roles": list(entry.roles),
        }
        for entry in sorted(catalog, key=lambda item: item.font_name.casefold())
    ]


def font_match_cache_key(
    catalog: Sequence[FontCatalogEntry],
    source_text: str,
    profile: Mapping[str, Any],
) -> str:
    payload = {
        "catalog": _catalog_contract(catalog),
        "profile": _profile_transform(profile),
        "source_text": str(source_text),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _shape_score(source: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    source_binary = source > 0
    candidate_binary = candidate > 0
    intersection = int(np.count_nonzero(source_binary & candidate_binary))
    union = int(np.count_nonzero(source_binary | candidate_binary))
    iou = intersection / max(1.0, float(union))
    source_distance = cv2.distanceTransform((~source_binary).astype(np.uint8), cv2.DIST_L2, 3)
    candidate_distance = cv2.distanceTransform((~candidate_binary).astype(np.uint8), cv2.DIST_L2, 3)
    forward = float(np.mean(candidate_distance[source_binary])) if np.any(source_binary) else 99.0
    backward = float(np.mean(source_distance[candidate_binary])) if np.any(candidate_binary) else 99.0
    chamfer = (forward + backward) / (2.0 * NORMALIZED_GLYPH_HEIGHT)
    contour_score = float(np.exp(-chamfer * 18.0))
    return 0.72 * iou + 0.28 * contour_score, iou, chamfer


class FontShapeMatcher:
    """Rank local fonts against a mask of the frozen source-language text."""

    def __init__(self, catalog: Sequence[FontCatalogEntry]) -> None:
        if not catalog:
            raise ValueError("font matcher requires a non-empty catalog")
        self.catalog = tuple(catalog)
        self.catalog_version = hashlib.sha256(
            _canonical_json(_catalog_contract(self.catalog))
        ).hexdigest()
        self._cache: dict[str, FontMatchResult] = {}

    def match(
        self,
        source_mask: np.ndarray,
        *,
        source_text: str,
        profile: Mapping[str, Any],
        semantic_role: str,
        translated_text: str | None = None,
        shortlist: Iterable[str] | None = None,
    ) -> FontMatchResult:
        del translated_text  # Translation must never influence source-font matching.
        normalized_source = _normalize_mask(source_mask)
        base_key = font_match_cache_key(self.catalog, source_text, profile)
        source_hash = hashlib.sha256(normalized_source.tobytes()).hexdigest()
        key = hashlib.sha256(
            _canonical_json({"base": base_key, "role": semantic_role, "source_mask": source_hash})
        ).hexdigest()
        result_metadata = {
            "catalog_version": self.catalog_version,
            "normalization": {
                "canvas": list(NORMALIZED_CANVAS),
                "glyph_height": NORMALIZED_GLYPH_HEIGHT,
                "profile": _profile_transform(profile),
            },
            "source_text_sha256": hashlib.sha256(
                str(source_text).encode("utf-8")
            ).hexdigest(),
            "glyph_mask_sha256": hashlib.sha256(
                np.ascontiguousarray(np.where(np.asarray(source_mask) > 0, 255, 0), dtype=np.uint8).tobytes()
            ).hexdigest(),
        }
        cached = self._cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        allowed = {str(value).casefold() for value in shortlist or ()}
        entries = [
            entry for entry in self.catalog
            if not allowed or entry.font_name.casefold() in allowed
        ]
        ranked: list[dict[str, Any]] = []
        for entry in entries:
            rendered = render_source_text_mask(source_text, entry.path, profile=profile)
            score, iou, chamfer = _shape_score(normalized_source, rendered)
            role_prior = 0.015 if semantic_role in entry.roles else 0.0
            ranked.append(
                {
                    "font_name": entry.font_name,
                    "score": round(min(1.0, score + role_prior), 6),
                    "silhouette_iou": round(iou, 6),
                    "normalized_chamfer": round(chamfer, 6),
                    "role_prior": role_prior,
                }
            )
        ranked.sort(key=lambda item: (-item["score"], item["font_name"].casefold()))
        if not ranked:
            result = FontMatchResult("unknown", "unknown", 0.0, 0.0, (), "empty_shortlist", key, **result_metadata)
        else:
            best = ranked[0]
            runner_up = ranked[1]["score"] if len(ranked) > 1 else 0.0
            margin = max(0.0, float(best["score"]) - float(runner_up))
            if float(best["score"]) < MATCH_SCORE_THRESHOLD:
                result = FontMatchResult(
                    "unknown", "unknown", 0.0, round(margin, 6), tuple(ranked[:5]),
                    "shape_score_below_threshold", key, **result_metadata,
                )
            elif margin < MATCH_MARGIN_THRESHOLD:
                result = FontMatchResult(
                    "unknown", "unknown", 0.0, round(margin, 6), tuple(ranked[:5]),
                    "insufficient_margin", key, **result_metadata,
                )
            else:
                confidence = min(1.0, float(best["score"]) * (0.75 + min(0.25, margin * 4.0)))
                result = FontMatchResult(
                    str(best["font_name"]), "exact", round(confidence, 6),
                    round(margin, 6), tuple(ranked[:5]), "", key, **result_metadata,
                )
        self._cache[key] = result
        return copy.deepcopy(result)
