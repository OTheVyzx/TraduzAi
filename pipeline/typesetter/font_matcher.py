"""Match source typography by rerendering frozen source text locally."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.textpath import TextPath


MATCH_SCORE_THRESHOLD = 0.68
MULTILINE_MATCH_SCORE_THRESHOLD = 0.58
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
    requested_map = Path(font_map_path).resolve()
    if (os.environ.get("TRADUZAI_QUALITY_CLOSED_FONTS") == "1"
            and requested_map.name == "font-map.json"):
        closed_path = Path(__file__).with_name("font-map.closed.json")
        closed = json.loads(closed_path.read_text(encoding="utf-8"))
        if closed.get("source_map_sha256") != _file_sha256(requested_map):
            raise ValueError("active font map changed since closed path binding")
        payload = closed
    else:
        payload = json.loads(requested_map.read_text(encoding="utf-8"))
    entries: list[FontCatalogEntry] = []
    seen_files: set[Path] = set()
    for raw in payload.get("available", []):
        if not isinstance(raw, Mapping) or not raw.get("detector", True):
            continue
        font_name = str(raw.get("arquivo") or "").strip()
        license_status = str(raw.get("license_status") or "").strip()
        roles = tuple(sorted({str(item).strip() for item in raw.get("roles", []) if str(item).strip()}))
        relative = Path(str(raw.get("path") or font_name))
        if not font_name or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid font map path: {font_name!r}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError(f"font catalog references missing file: {font_name!r}")
        if not license_status or not roles:
            raise ValueError(f"font catalog entry lacks license/role metadata: {font_name}")
        if resolved in seen_files:
            continue
        seen_files.add(resolved)
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
    canvas_height, canvas_width = NORMALIZED_CANVAS
    scale = min(
        NORMALIZED_GLYPH_HEIGHT / max(1.0, float(height)),
        (canvas_width - 8) / max(1.0, float(width)),
    )
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros(NORMALIZED_CANVAS, dtype=np.uint8)
    top = (canvas_height - resized.shape[0]) // 2
    left = (canvas_width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _render_text_path_raw_mask(text: str, font_path: Path) -> np.ndarray:
    """Render one source line at its natural TextPath geometry."""

    if not str(text or "").strip():
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
    return mask


def _apply_profile_transform(mask: np.ndarray, profile: Mapping[str, Any]) -> np.ndarray:
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
    return mask


def render_source_text_mask(
    source_text: str,
    font_path: Path,
    *,
    profile: Mapping[str, Any],
) -> np.ndarray:
    """Render frozen source text and apply measured geometric transforms."""

    mask = _render_text_path_raw_mask(str(source_text or ""), font_path)
    mask = _apply_profile_transform(mask, profile)
    return _normalize_mask(mask)


def render_source_text_lines_mask(
    lines: Sequence[str],
    font_path: Path,
    *,
    profile: Mapping[str, Any],
    normalized_gap: int = 14,
) -> np.ndarray:
    """Render source lines at one natural scale before block normalization."""

    raw_crops: list[np.ndarray] = []
    for line in lines:
        raw = _render_text_path_raw_mask(str(line), font_path)
        points = cv2.findNonZero(raw)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        raw_crops.append(raw[y : y + height, x : x + width])
    if not raw_crops:
        raise ValueError("font produced no renderable source lines")
    median_height = max(1.0, float(np.median([crop.shape[0] for crop in raw_crops])))
    raw_gap = max(1, int(round(normalized_gap / NORMALIZED_GLYPH_HEIGHT * median_height)))
    canvas = np.zeros(
        (
            sum(crop.shape[0] for crop in raw_crops) + raw_gap * (len(raw_crops) - 1),
            max(crop.shape[1] for crop in raw_crops),
        ),
        dtype=np.uint8,
    )
    cursor_y = 0
    for crop in raw_crops:
        left = (canvas.shape[1] - crop.shape[1]) // 2
        canvas[cursor_y : cursor_y + crop.shape[0], left : left + crop.shape[1]] = crop
        cursor_y += crop.shape[0] + raw_gap
    return _normalize_mask(_apply_profile_transform(canvas, profile))


def _source_line_masks(
    mask: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...], int]:
    """Separate glyph components into source lines and measure their layout."""

    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None:
        return ((binary,), (1.0,), 14)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        connectivity=8,
    )
    components = [
        {
            "label": label,
            "x1": int(stats[label, cv2.CC_STAT_LEFT]),
            "y1": int(stats[label, cv2.CC_STAT_TOP]),
            "x2": int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
            "y2": int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
            "height": int(stats[label, cv2.CC_STAT_HEIGHT]),
            "area": int(stats[label, cv2.CC_STAT_AREA]),
            "center_y": float(centroids[label, 1]),
        }
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 4
    ]
    if len(components) < 2:
        return ((binary,), (1.0,), 14)
    median_area = max(1.0, float(np.median([item["area"] for item in components])))
    substantial = [
        item for item in components
        if item["area"] >= max(4.0, median_area * 0.18)
    ]
    if len(substantial) < 2:
        return ((binary,), (1.0,), 14)
    median_height = max(1.0, float(np.median([item["height"] for item in substantial])))
    center_threshold = max(2.0, median_height * 0.72)
    clusters: list[list[dict[str, float | int]]] = []
    for component in sorted(substantial, key=lambda item: (item["center_y"], item["x1"])):
        if not clusters:
            clusters.append([component])
            continue
        current_center = float(np.median([item["center_y"] for item in clusters[-1]]))
        if abs(float(component["center_y"]) - current_center) <= center_threshold:
            clusters[-1].append(component)
        else:
            clusters.append([component])
    if len(clusters) <= 1:
        return ((binary,), (1.0,), 14)

    assigned_labels = {
        int(item["label"])
        for cluster in clusters
        for item in cluster
    }
    cluster_centers = [
        float(np.median([item["center_y"] for item in cluster]))
        for cluster in clusters
    ]
    for component in components:
        if int(component["label"]) in assigned_labels:
            continue
        distances = [
            abs(float(component["center_y"]) - center)
            for center in cluster_centers
        ]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= median_height * 1.25:
            clusters[nearest].append(component)

    line_masks = tuple(
        np.where(
            np.isin(labels, [int(item["label"]) for item in cluster]),
            255,
            0,
        ).astype(np.uint8)
        for cluster in clusters
    )

    widths = [
        float(max(item["x2"] for item in cluster) - min(item["x1"] for item in cluster))
        for cluster in clusters
    ]
    maximum_width = max(widths)
    width_ratios = tuple(width / maximum_width for width in widths)
    centers = [
        float(np.median([item["center_y"] for item in cluster]))
        for cluster in clusters
    ]
    center_steps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    median_gap = max(
        1.0,
        (float(np.median(center_steps)) - median_height) if center_steps else median_height * 0.11,
    )
    normalized_gap = int(
        round(np.clip(median_gap / median_height * NORMALIZED_GLYPH_HEIGHT, 8, 48))
    )
    return (line_masks, width_ratios, normalized_gap)


def _partition_source_text(
    source_text: str,
    font_path: Path,
    target_width_ratios: Sequence[float],
) -> tuple[str, ...]:
    """Split frozen source words to match measured source-line proportions."""

    words = str(source_text or "").split()
    line_count = min(len(words), len(target_width_ratios))
    if line_count <= 1:
        return (" ".join(words),)
    target_total = max(1e-6, float(sum(target_width_ratios[:line_count])))
    target_shares = [float(value) / target_total for value in target_width_ratios[:line_count]]
    font = FT2Font(str(font_path))
    font.set_size(112, 72)

    def text_width(text: str) -> float:
        font.set_text(str(text), 0.0)
        return max(1.0, float(font.get_width_height()[0]) / 64.0)

    word_widths = [text_width(word) for word in words]
    reference_width = text_width("M")
    space_width = max(
        1.0,
        text_width("M M") - reference_width * 2.0,
    )
    prefix_widths = [0.0]
    for word_width in word_widths:
        prefix_widths.append(prefix_widths[-1] + word_width)
    full_width = prefix_widths[-1] + space_width * max(0, len(words) - 1)

    def segment_width(start: int, end: int) -> float:
        return (
            prefix_widths[end]
            - prefix_widths[start]
            + space_width * max(0, end - start - 1)
        )

    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {(0, 0): (0.0, ())}
    for line_index in range(line_count):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (_previous_line, start), (cost, breaks) in states.items():
            remaining_lines = line_count - line_index - 1
            maximum_end = len(words) - remaining_lines
            for end in range(start + 1, maximum_end + 1):
                share = segment_width(start, end) / full_width
                candidate = (
                    cost + (share - target_shares[line_index]) ** 2,
                    (*breaks, end),
                )
                key = (line_index + 1, end)
                current = next_states.get(key)
                if current is None or candidate < current:
                    next_states[key] = candidate
        states = next_states
    _cost, breaks = states[(line_count, len(words))]
    lines: list[str] = []
    start = 0
    for end in breaks:
        lines.append(" ".join(words[start:end]))
        start = end
    return tuple(lines)


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
        "policy": {"metric": "shape_score_v2_multiline",
                   "single_threshold": MATCH_SCORE_THRESHOLD,
                   "multiline_threshold": MULTILINE_MATCH_SCORE_THRESHOLD,
                   "margin_threshold": MATCH_MARGIN_THRESHOLD},
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


def _intrinsic_slant_tangent(mask: np.ndarray) -> float:
    """Estimate glyph slant from per-component row-centroid drift."""

    binary = np.where(np.asarray(mask) > 0, 1, 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= 4
    ]
    if not areas:
        return 0.0
    minimum_area = max(8.0, float(np.median(areas)) * 0.15)
    tangents: list[float] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area or height < 5:
            continue
        component = labels[y : y + height, x : x + width] == label
        row_y: list[float] = []
        row_x: list[float] = []
        for offset_y in range(height):
            occupied_x = np.flatnonzero(component[offset_y])
            if occupied_x.size:
                row_y.append(float(offset_y))
                row_x.append(float(np.mean(occupied_x)))
        if len(row_y) >= 5:
            tangents.append(float(np.polyfit(row_y, row_x, 1)[0]))
    return float(np.median(tangents)) if tangents else 0.0


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
        ranking_limit: int = 5,
    ) -> FontMatchResult:
        del translated_text  # Translation must never influence source-font matching.
        normalized_source = _normalize_mask(source_mask)
        source_line_masks, source_line_width_ratios, normalized_line_gap = _source_line_masks(
            source_mask
        )
        base_key = font_match_cache_key(self.catalog, source_text, profile)
        source_hash = hashlib.sha256(normalized_source.tobytes()).hexdigest()
        key = hashlib.sha256(
            _canonical_json({"base": base_key, "role": semantic_role, "source_mask": source_hash,
                             "ranking_limit": ranking_limit})
        ).hexdigest()
        result_metadata = {
            "catalog_version": self.catalog_version,
            "normalization": {
                "canvas": list(NORMALIZED_CANVAS),
                "glyph_height": NORMALIZED_GLYPH_HEIGHT,
                "profile": _profile_transform(profile),
                "source_line_count": len(source_line_width_ratios),
                "source_line_width_ratios": [
                    round(value, 6) for value in source_line_width_ratios
                ],
                "normalized_line_gap": normalized_line_gap,
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
        source_line_slants = [
            _intrinsic_slant_tangent(line_mask)
            for line_mask in source_line_masks
        ]
        for entry in entries:
            candidate_lines = _partition_source_text(
                source_text,
                entry.path,
                source_line_width_ratios,
            )
            if len(source_line_masks) > 1 and len(candidate_lines) == len(source_line_masks):
                line_metrics: list[tuple[float, float, float, float, float]] = []
                for line_index, (source_line_mask, candidate_line) in enumerate(zip(
                    source_line_masks,
                    candidate_lines,
                    strict=True,
                )):
                    normalized_line = _normalize_mask(source_line_mask)
                    rendered_line = render_source_text_mask(
                        candidate_line,
                        entry.path,
                        profile=profile,
                    )
                    line_score, line_iou, line_chamfer = _shape_score(
                        normalized_line,
                        rendered_line,
                    )
                    candidate_slant = _intrinsic_slant_tangent(rendered_line)
                    slant_similarity = float(
                        np.exp(
                            -abs(source_line_slants[line_index] - candidate_slant)
                            / 0.08
                        )
                    )
                    line_score = 0.82 * line_score + 0.18 * slant_similarity
                    weight = float(np.count_nonzero(source_line_mask))
                    line_metrics.append(
                        (line_score, line_iou, line_chamfer, slant_similarity, weight)
                    )
                total_weight = max(1.0, sum(item[4] for item in line_metrics))
                score = sum(item[0] * item[4] for item in line_metrics) / total_weight
                iou = sum(item[1] * item[4] for item in line_metrics) / total_weight
                chamfer = sum(item[2] * item[4] for item in line_metrics) / total_weight
                slant_similarity = (
                    sum(item[3] * item[4] for item in line_metrics) / total_weight
                )
            else:
                rendered = render_source_text_mask(source_text, entry.path, profile=profile)
                score, iou, chamfer = _shape_score(normalized_source, rendered)
                slant_similarity = float(
                    np.exp(
                        -abs(
                            _intrinsic_slant_tangent(normalized_source)
                            - _intrinsic_slant_tangent(rendered)
                        )
                        / 0.08
                    )
                )
            role_prior = 0.015 if semantic_role in entry.roles else 0.0
            ranked.append(
                {
                    "font_name": entry.font_name,
                    "rasterized_sample": candidate_lines if len(source_line_masks)>1
                    and len(candidate_lines)==len(source_line_masks) else [source_text],
                    "score": round(min(1.0, score + role_prior), 6),
                    "silhouette_iou": round(iou, 6),
                    "normalized_chamfer": round(chamfer, 6),
                    "slant_similarity": round(slant_similarity, 6),
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
            score_threshold = (
                MULTILINE_MATCH_SCORE_THRESHOLD
                if len(source_line_masks) > 1
                else MATCH_SCORE_THRESHOLD
            )
            if float(best["score"]) < score_threshold:
                result = FontMatchResult(
                    "unknown", "unknown", 0.0, round(margin, 6), tuple(ranked[:ranking_limit]),
                    "shape_score_below_threshold", key, **result_metadata,
                )
            elif margin < MATCH_MARGIN_THRESHOLD:
                result = FontMatchResult(
                    "unknown", "unknown", 0.0, round(margin, 6), tuple(ranked[:ranking_limit]),
                    "insufficient_margin", key, **result_metadata,
                )
            else:
                if len(source_line_masks) > 1:
                    score_headroom = max(
                        0.0,
                        (float(best["score"]) - score_threshold)
                        / max(1e-6, 1.0 - score_threshold),
                    )
                    confidence = min(
                        1.0,
                        0.70
                        + min(0.20, score_headroom * 0.20)
                        + min(0.10, margin * 2.0),
                    )
                else:
                    confidence = min(1.0, float(best["score"]) * (0.75 + min(0.25, margin * 4.0)))
                result = FontMatchResult(
                    str(best["font_name"]), "exact", round(confidence, 6),
                    round(margin, 6), tuple(ranked[:ranking_limit]), "", key, **result_metadata,
                )
        self._cache[key] = result
        return copy.deepcopy(result)
