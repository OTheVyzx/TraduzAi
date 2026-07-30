"""Page-global discovery of visual source-text evidence before OCR acceptance.

The scanner deliberately reasons about geometry and contrast only.  Recognized
text is not an input: OCR is allowed to explain a component later, but it is
not allowed to decide whether the source pixels existed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np

from .coordinates import ComponentSeed, assign_component_ids
from .model import BBox, Point, SourceTextComponent


Polygon = tuple[Point, ...]
_MAX_STROKE_COMPONENTS = 4096
_MAX_GLYPH_CANDIDATES = 256


@dataclass(frozen=True)
class DetectorRegion:
    """Text-region evidence produced by a detector, expressed on the page."""

    bbox_page: BBox
    polygon_page: Polygon = ()
    detector_source: str = "region_detector"
    confidence: float = 1.0
    evidence_id: str | None = None
    script_evidence: tuple[str, ...] = ()
    rotation_deg: float | None = None
    rotation_source: str | None = None
    support_only: bool = False


@dataclass(frozen=True)
class GlyphCandidate:
    """Textlike stroke evidence produced independently from OCR recognition."""

    bbox_page: BBox
    polygon_page: Polygon = ()
    detector_source: str = "glyph_scan"
    confidence: float = 1.0
    evidence_id: str | None = None
    script_evidence: tuple[str, ...] = ()
    rotation_deg: float | None = None
    rotation_source: str | None = None


@dataclass(frozen=True)
class _Evidence:
    bbox_page: BBox
    polygon_page: Polygon
    detector_sources: tuple[str, ...]
    confidence: float
    evidence_ids: tuple[str, ...] = ()
    script_evidence: tuple[str, ...] = ()
    rotation_deg: float | None = None
    rotation_source: str | None = None
    support_only: bool = False


@dataclass(frozen=True)
class _GlyphStructure:
    count: int
    foreground_ratio: float
    oriented_fill_ratio: float
    mean_component_fill: float
    mean_contour_vertices: float
    linearity: float
    mean_circularity: float
    hole_fraction: float


@dataclass(frozen=True)
class _StrokePrimitive:
    label: int
    bbox_page: BBox
    center: tuple[float, float]
    area: int


def _bbox_area(bbox: BBox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_intersection(left: BBox, right: BBox) -> int:
    return (
        max(0, min(left[2], right[2]) - max(left[0], right[0]))
        * max(0, min(left[3], right[3]) - max(left[1], right[1]))
    )


def _bbox_iou(left: BBox, right: BBox) -> float:
    intersection = _bbox_intersection(left, right)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(left) + _bbox_area(right) - intersection
    return intersection / float(max(1, union))


def _bbox_union(boxes: Iterable[BBox]) -> BBox:
    materialised = list(boxes)
    return (
        min(box[0] for box in materialised),
        min(box[1] for box in materialised),
        max(box[2] for box in materialised),
        max(box[3] for box in materialised),
    )


def _rect_polygon(bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = bbox
    return (x1, y1), (x2, y1), (x2, y2), (x1, y2)


def _clamp_bbox(bbox: Sequence[int], *, width: int, height: int) -> BBox | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _normalise_polygon(
    points: Iterable[Sequence[int]],
    bbox: BBox,
    *,
    width: int,
    height: int,
) -> Polygon:
    polygon: list[Point] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = max(0, min(width, int(round(float(point[0])))))
            y = max(0, min(height, int(round(float(point[1])))))
        except (TypeError, ValueError):
            continue
        polygon.append((x, y))
    return tuple(polygon) if len(polygon) >= 3 else _rect_polygon(bbox)


def _polygon_hull(polygons: Iterable[Polygon], fallback_bbox: BBox) -> Polygon:
    points = [point for polygon in polygons for point in polygon]
    if len(points) < 3:
        return _rect_polygon(fallback_bbox)
    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
    if hull is None or len(hull) < 3:
        return _rect_polygon(fallback_bbox)
    return tuple((int(point[0][0]), int(point[0][1])) for point in hull)


def _gray_image(page_rgb: np.ndarray) -> np.ndarray:
    if page_rgb.ndim == 2:
        return page_rgb.astype(np.uint8, copy=False)
    return cv2.cvtColor(page_rgb[:, :, :3], cv2.COLOR_RGB2GRAY)


def _has_meaningful_chroma(page_rgb: np.ndarray) -> bool:
    if page_rgb.ndim != 3 or page_rgb.shape[2] < 3:
        return False
    rgb = page_rgb[:, :, :3].astype(np.int16, copy=False)
    spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return float(np.percentile(spread, 90.0)) >= 3.0


def _contrast_mask(page_image: np.ndarray) -> np.ndarray:
    """Return locally salient strokes without assuming a text luminance.

    Manga lettering is frequently coloured so that foreground and background
    have almost identical grayscale luminance.  Work on every RGB channel (or
    the sole grayscale channel) and at two spatial scales, then derive the
    threshold from the page noise floor.  The bounded threshold deliberately
    keeps faint antialiased glyph edges while the later structural classifier
    decides whether those edges form text.
    """

    if page_image.ndim == 2:
        channels = (page_image.astype(np.uint8, copy=False),)
    else:
        rgb = page_image[:, :, :3].astype(np.uint8, copy=False)
        channels = tuple(cv2.split(rgb)) if _has_meaningful_chroma(rgb) else (_gray_image(rgb),)

    responses: list[np.ndarray] = []
    for channel in channels:
        for sigma in (1.4, 3.5):
            local_background = cv2.GaussianBlur(
                channel,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )
            responses.append(cv2.absdiff(channel, local_background))
    difference = np.maximum.reduce(responses)
    median = float(np.median(difference))
    mad = float(np.median(np.abs(difference.astype(np.float32) - median)))
    threshold = int(round(np.clip(median + 3.5 * max(1.0, mad), 4.0, 14.0)))
    mask = (difference >= threshold).astype(np.uint8) * 255
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )


def _candidate_stroke_boxes(
    page_rgb: np.ndarray,
    *,
    gray: np.ndarray | None = None,
    strokes: np.ndarray | None = None,
) -> list[BBox]:
    if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
        return []
    gray = _gray_image(page_rgb) if gray is None else gray
    height, width = gray.shape[:2]
    page_area = max(1, height * width)
    strokes = _contrast_mask(page_rgb) if strokes is None else strokes

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(strokes, connectivity=8)
    if count - 1 > _MAX_STROKE_COMPONENTS:
        # Dense texture is not safe evidence.  Region detectors remain available
        # as the conservative fallback, and the scan stays within a page budget.
        return []

    boxes: list[BBox] = []
    max_area = max(480, int(page_area * 0.14))
    max_width = max(48, int(width * 0.88))
    max_height = max(28, int(height * 0.88))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > max_area:
            continue
        if box_width < 2 or box_height < 3:
            continue
        if box_width > max_width or box_height > max_height:
            continue
        aspect = max(box_width, box_height) / float(max(1, min(box_width, box_height)))
        fill = area / float(max(1, box_width * box_height))
        if aspect > 20.0 or fill < 0.025 or fill > 0.97:
            continue
        boxes.append((x, y, x + box_width, y + box_height))
    return boxes


def _polarity_candidate_boxes(page_rgb: np.ndarray) -> list[BBox]:
    """Find word/line blobs that local-difference CCs fragment into letters.

    Black-hat and top-hat responses suppress slowly varying balloon/card
    texture while retaining dark-on-light and light-on-dark glyph groups.
    These boxes still pass the same glyph-structure checks as the primary
    scan; morphology alone never promotes a source component.
    """

    gray = _gray_image(page_rgb)
    height, width = gray.shape[:2]
    page_area = max(1, height * width)
    if page_rgb.ndim == 2 or not _has_meaningful_chroma(page_rgb):
        channels = (gray,)
    else:
        rgb = page_rgb[:, :, :3].astype(np.uint8, copy=False)
        channels = tuple(cv2.split(rgb))
    kernel_sizes = (9, 17) if min(height, width) >= 160 else (7, 11)
    boxes: list[BBox] = []
    for channel in channels:
        for kernel_size in kernel_sizes:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size),
            )
            responses = (
                cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel),
                cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel),
            )
            for response in responses:
                median = float(np.median(response))
                mad = float(
                    np.median(np.abs(response.astype(np.float32) - median))
                )
                threshold = int(
                    round(np.clip(median + 4.0 * max(1.0, mad), 4.0, 32.0))
                )
                mask = (response >= threshold).astype(np.uint8) * 255
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
                    iterations=1,
                )
                count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8,
                )
                if count - 1 > _MAX_STROKE_COMPONENTS:
                    continue
                for label in range(1, count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    x = int(stats[label, cv2.CC_STAT_LEFT])
                    y = int(stats[label, cv2.CC_STAT_TOP])
                    box_width = int(stats[label, cv2.CC_STAT_WIDTH])
                    box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
                    if area < 24 or area > page_area * 0.12:
                        continue
                    if box_width < 6 or box_height < 6:
                        continue
                    if (
                        x <= 1
                        or y <= 1
                        or x + box_width >= width - 1
                        or y + box_height >= height - 1
                    ):
                        continue
                    if box_width > width * 0.96 or box_height > height * 0.45:
                        continue
                    aspect = max(box_width, box_height) / float(
                        max(1, min(box_width, box_height))
                    )
                    fill = area / float(max(1, box_width * box_height))
                    if aspect > 18.0 or fill < 0.05 or fill > 0.92:
                        continue
                    boxes.append((x, y, x + box_width, y + box_height))

    deduped: list[BBox] = []
    for bbox in sorted(boxes, key=lambda item: (item[1], item[0], item[3], item[2])):
        if any(_bbox_iou(bbox, existing) >= 0.82 for existing in deduped):
            continue
        deduped.append(bbox)
    return deduped


def _local_foreground_mask(page_rgb: np.ndarray) -> np.ndarray:
    """Separate local ink from its border-estimated surface in colour space."""

    if page_rgb.size == 0:
        return np.zeros(page_rgb.shape[:2], dtype=np.uint8)
    if page_rgb.ndim == 2:
        channels = page_rgb[:, :, None].astype(np.float32)
    else:
        rgb = page_rgb[:, :, :3].astype(np.uint8, copy=False)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        channels = np.dstack((rgb, lab[:, :, 1:])).astype(np.float32)
    border = np.concatenate(
        (
            channels[0, :, :],
            channels[-1, :, :],
            channels[:, 0, :],
            channels[:, -1, :],
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.max(np.abs(channels - background), axis=2)
    if float(np.max(distance)) < 3.0:
        return np.zeros(distance.shape, dtype=np.uint8)
    distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    threshold, _unused = cv2.threshold(
        distance_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = max(3.0, float(threshold))
    return (distance >= threshold).astype(np.uint8) * 255


def _minority_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Return the locally minority luminance class for ordinary lettering."""

    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    threshold, _unused = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    border = np.concatenate(
        (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1])
    )
    background = float(np.median(border)) if border.size else float(np.median(gray))
    if background >= float(threshold):
        return (gray <= threshold).astype(np.uint8) * 255
    return (gray > threshold).astype(np.uint8) * 255


def _internal_glyph_structure(page_rgb: np.ndarray, bbox: BBox) -> _GlyphStructure:
    x1, y1, x2, y2 = bbox
    crop = page_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return _GlyphStructure(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    gray = _gray_image(crop)
    # Luminance segmentation is more stable for ordinary black/white glyphs.
    # Fall back to colour distance only when luminance genuinely carries no
    # separation, so equal-luma lettering remains discoverable.
    gray_range = int(np.max(gray)) - int(np.min(gray))
    colour_range = gray_range
    if crop.ndim == 3 and crop.shape[2] >= 3:
        colour_range = max(
            int(np.max(channel)) - int(np.min(channel))
            for channel in cv2.split(crop[:, :, :3])
        )
    luminance_is_weak = gray_range <= max(6.0, 0.08 * colour_range)
    mask = (
        _local_foreground_mask(crop)
        if luminance_is_weak
        else _minority_foreground_mask(gray)
    )
    roi_area = max(1, mask.shape[0] * mask.shape[1])
    foreground_ratio = cv2.countNonZero(mask) / float(roi_area)
    if foreground_ratio < 0.01 or foreground_ratio > 0.78:
        return _GlyphStructure(0, foreground_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def parts(
        candidate_mask: np.ndarray,
    ) -> tuple[
        list[tuple[float, float]],
        list[float],
        list[float],
        list[float],
        list[bool],
    ]:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8,
        )
        minimum_area = max(3, int(round(roi_area * 0.0015)))
        centroids: list[tuple[float, float]] = []
        circularities: list[float] = []
        component_fills: list[float] = []
        contour_vertices: list[float] = []
        has_holes: list[bool] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            part_width = int(stats[label, cv2.CC_STAT_WIDTH])
            part_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < minimum_area or area > roi_area * 0.65:
                continue
            if part_width < 2 or part_height < 3:
                continue
            centroids.append(
                (float(_centroids[label][0]), float(_centroids[label][1]))
            )
            component_fills.append(area / float(max(1, part_width * part_height)))
            component_mask = (_labels == label).astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                component_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                external_indexes = [
                    index
                    for index in range(len(contours))
                    if hierarchy is None or int(hierarchy[0, index, 3]) < 0
                ]
                contour = max(
                    (contours[index] for index in external_indexes),
                    key=cv2.contourArea,
                )
                perimeter = float(cv2.arcLength(contour, True))
                contour_area = float(cv2.contourArea(contour))
                circularities.append(
                    (4.0 * np.pi * contour_area / (perimeter * perimeter))
                    if perimeter > 0.0
                    else 0.0
                )
                approximation = cv2.approxPolyDP(
                    contour,
                    0.03 * perimeter,
                    True,
                )
                contour_vertices.append(float(len(approximation)))
                has_holes.append(
                    bool(
                        hierarchy is not None
                        and any(int(hierarchy[0, index, 3]) >= 0 for index in range(len(contours)))
                    )
                )
            else:
                circularities.append(0.0)
                contour_vertices.append(0.0)
                has_holes.append(False)
        return centroids, circularities, component_fills, contour_vertices, has_holes

    centroids, circularities, component_fills, contour_vertices, has_holes = parts(mask)
    if len(centroids) < 2:
        eroded = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )
        (
            eroded_centroids,
            eroded_circularities,
            eroded_fills,
            eroded_vertices,
            eroded_holes,
        ) = parts(eroded)
        if len(eroded_centroids) >= 3 and len(eroded_centroids) >= len(centroids) + 2:
            centroids, circularities, component_fills, contour_vertices, has_holes = (
                eroded_centroids,
                eroded_circularities,
                eroded_fills,
                eroded_vertices,
                eroded_holes,
            )
    glyph_count = min(len(centroids), 65)
    linearity = 0.0
    if glyph_count == 2:
        linearity = 1.0
    elif glyph_count >= 3:
        points = np.asarray(centroids, dtype=np.float64)
        covariance = np.cov(points.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        linearity = float(eigenvalues[-1] / max(1e-6, float(eigenvalues.sum())))
    foreground_points = cv2.findNonZero(mask)
    oriented_area = 0.0
    if foreground_points is not None and len(foreground_points) >= 3:
        rect_width, rect_height = cv2.minAreaRect(foreground_points)[1]
        oriented_area = float(rect_width) * float(rect_height)
    return _GlyphStructure(
        count=glyph_count,
        foreground_ratio=foreground_ratio,
        oriented_fill_ratio=(cv2.countNonZero(mask) / max(1.0, oriented_area)),
        mean_component_fill=(
            float(np.mean(component_fills)) if component_fills else 0.0
        ),
        mean_contour_vertices=(
            float(np.mean(contour_vertices)) if contour_vertices else 0.0
        ),
        linearity=linearity,
        mean_circularity=(float(np.mean(circularities)) if circularities else 0.0),
        hole_fraction=(float(np.mean(has_holes)) if has_holes else 0.0),
    )


def _contrast_polygon(strokes_page: np.ndarray, bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = bbox
    strokes = strokes_page[y1:y2, x1:x2]
    if strokes.size == 0:
        return _rect_polygon(bbox)
    points = cv2.findNonZero(strokes)
    if points is None or len(points) < 3:
        return _rect_polygon(bbox)
    hull = cv2.convexHull(points, clockwise=False, returnPoints=True)
    return tuple(
        (int(point[0][0]) + x1, int(point[0][1]) + y1)
        for point in hull
    )


def _stroke_primitive_records(
    strokes: np.ndarray,
) -> tuple[list[_StrokePrimitive], np.ndarray]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        strokes,
        connectivity=8,
    )
    result: list[_StrokePrimitive] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < 3 or box_width < 2 or box_height < 3:
            continue
        result.append(
            _StrokePrimitive(
                label=label,
                bbox_page=(x, y, x + box_width, y + box_height),
                center=(float(centroids[label][0]), float(centroids[label][1])),
                area=area,
            )
        )
    return result, labels


def _local_primitive_count(
    primitives: Sequence[_StrokePrimitive],
    bbox: BBox,
    *,
    width: int,
    height: int,
) -> int:
    """Count nearby visual strokes without consulting the rest of the page."""

    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    local = (
        max(0.0, center_x - max(64.0, 2.5 * box_width)),
        max(0.0, center_y - max(64.0, 2.5 * box_height)),
        min(float(width), center_x + max(64.0, 2.5 * box_width)),
        min(float(height), center_y + max(64.0, 2.5 * box_height)),
    )
    return sum(
        local[0] <= item.center[0] <= local[2]
        and local[1] <= item.center[1] <= local[3]
        for item in primitives
    )


def _attached_to_large_visual_edge(
    primitives: Sequence[_StrokePrimitive],
    labels: np.ndarray,
    bbox: BBox,
) -> bool:
    """Detect a proposal dominated by a small fragment of a much larger edge."""

    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    box_area = box_width * box_height
    label_crop = labels[y1:y2, x1:x2]
    if label_crop.size == 0:
        return False
    total_stroke_pixels = int(np.count_nonzero(label_crop))
    if total_stroke_pixels <= 0:
        return False
    for item in primitives:
        component_width = item.bbox_page[2] - item.bbox_page[0]
        component_height = item.bbox_page[3] - item.bbox_page[1]
        is_much_larger = (
            component_width >= 4 * box_width
            or component_height >= 4 * box_height
            or item.area >= 12 * box_area
        )
        if not is_much_larger:
            continue
        attached_pixels = int(np.count_nonzero(label_crop == item.label))
        dominance = attached_pixels / float(total_stroke_pixels)
        primitive_ownership = attached_pixels / float(max(1, item.area))
        edge_fragment_score = dominance * (1.0 - primitive_ownership)
        if edge_fragment_score >= 0.80:
            return True
    return False


def _axis_overlap(left1: int, left2: int, right1: int, right2: int) -> int:
    return max(0, min(left2, right2) - max(left1, right1))


def _text_blobs_are_neighbours(
    left: BBox,
    right: BBox,
    *,
    left_glyph_count: int = 0,
    right_glyph_count: int = 0,
) -> bool:
    intersection = _bbox_intersection(left, right)
    if intersection > 0:
        left_area = max(1, _bbox_area(left))
        right_area = max(1, _bbox_area(right))
        smaller_area = min(left_area, right_area)
        larger_area = max(left_area, right_area)
        # A broad illustration/panel proposal can geometrically contain a
        # legitimate tight text line.  Treating every intersection as textual
        # affinity lets that art proposal swallow the line and the resulting
        # page-sized union is discarded later.  Comparable-scale proposals
        # still merge, while asymmetric containment remains independent.
        larger_glyph_count = (
            left_glyph_count if left_area >= right_area else right_glyph_count
        )
        smaller_glyph_count = (
            right_glyph_count if left_area >= right_area else left_glyph_count
        )
        if (
            larger_area / float(smaller_area) >= 8.0
            and intersection / float(smaller_area) >= 0.75
            and larger_glyph_count <= 3
            and smaller_glyph_count >= 4
        ):
            return False
        return True
    left_width, left_height = left[2] - left[0], left[3] - left[1]
    right_width, right_height = right[2] - right[0], right[3] - right[1]
    x_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]))
    y_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]))
    x_overlap = _axis_overlap(left[0], left[2], right[0], right[2])
    y_overlap = _axis_overlap(left[1], left[3], right[1], right[3])
    horizontal_scale = max(3.0, 0.90 * max(left_height, right_height))
    if y_overlap >= 0.15 * min(left_height, right_height) and x_gap <= horizontal_scale:
        return True
    both_vertical = (
        left_height >= 1.20 * left_width
        and right_height >= 1.20 * right_width
    )
    if both_vertical:
        vertical_scale = max(3.0, 0.90 * max(left_width, right_width))
        return (
            x_overlap >= 0.30 * min(left_width, right_width)
            and y_gap <= vertical_scale
        )
    return False


def _group_textlike_blobs(blobs: Sequence[tuple[BBox, int]]) -> list[list[tuple[BBox, int]]]:
    parents = list(range(len(blobs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left_index in range(len(blobs)):
        for right_index in range(left_index + 1, len(blobs)):
            left_bbox, left_glyph_count = blobs[left_index]
            right_bbox, right_glyph_count = blobs[right_index]
            if _text_blobs_are_neighbours(
                left_bbox,
                right_bbox,
                left_glyph_count=left_glyph_count,
                right_glyph_count=right_glyph_count,
            ):
                union(left_index, right_index)

    grouped: dict[int, list[tuple[BBox, int]]] = {}
    for index, blob in enumerate(blobs):
        grouped.setdefault(find(index), []).append(blob)
    return list(grouped.values())


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values)) / max(1e-6, abs(mean))


def _looks_periodic(group: Sequence[tuple[BBox, int]]) -> bool:
    if len(group) < 8:
        return False
    boxes = [item[0] for item in group]
    widths = [float(box[2] - box[0]) for box in boxes]
    heights = [float(box[3] - box[1]) for box in boxes]
    centers = sorted((box[0] + box[2]) / 2.0 for box in boxes)
    gaps = [right - left for left, right in zip(centers, centers[1:]) if right > left]
    return (
        _coefficient_of_variation(widths) <= 0.08
        and _coefficient_of_variation(heights) <= 0.08
        and len(gaps) >= 7
        and _coefficient_of_variation(gaps) <= 0.08
    )


def scan_textlike_glyph_candidates(page_rgb: np.ndarray) -> list[GlyphCandidate]:
    """Return bounded orientation-independent candidates from visual contrast."""

    if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
        return []
    height, width = page_rgb.shape[:2]
    gray = _gray_image(page_rgb)
    strokes = _contrast_mask(page_rgb)
    stroke_primitives, stroke_labels = _stroke_primitive_records(strokes)
    stroke_boxes = _candidate_stroke_boxes(page_rgb, gray=gray, strokes=strokes)
    for polarity_bbox in _polarity_candidate_boxes(page_rgb):
        if not any(_bbox_iou(polarity_bbox, existing) >= 0.82 for existing in stroke_boxes):
            stroke_boxes.append(polarity_bbox)
    textlike_structures: list[tuple[BBox, _GlyphStructure]] = []
    for bbox in stroke_boxes:
        structure = _internal_glyph_structure(page_rgb, bbox)
        box_width, box_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        aspect = max(box_width, box_height) / float(max(1, min(box_width, box_height)))
        if structure.count < 2:
            continue
        if structure.count == 2:
            if structure.oriented_fill_ratio < 0.24:
                continue
            if structure.mean_component_fill < 0.18:
                continue
            if structure.mean_contour_vertices < 6.0:
                continue
        else:
            if structure.oriented_fill_ratio < 0.13:
                continue
            if structure.mean_component_fill < 0.20:
                continue
            if structure.linearity < 0.70:
                continue
            if structure.mean_contour_vertices < 4.0:
                continue
        if structure.mean_component_fill > 0.78:
            continue
        if structure.count <= 3 and structure.mean_circularity >= 0.54:
            continue
        if structure.mean_circularity >= 0.70:
            continue
        if structure.hole_fraction >= 0.70 and structure.mean_circularity >= 0.35:
            continue
        if min(box_width, box_height) < 6 or aspect > 10.0:
            continue
        textlike_structures.append((bbox, structure))
        if len(textlike_structures) >= _MAX_GLYPH_CANDIDATES:
            break

    textlike_blobs = [
        (bbox, structure.count)
        for bbox, structure in textlike_structures
    ]

    candidates: list[GlyphCandidate] = []
    for group in _group_textlike_blobs(textlike_blobs):
        if _looks_periodic(group):
            continue
        bbox = _bbox_union(item[0] for item in group)
        bbox_area = _bbox_area(bbox)
        if bbox[2] - bbox[0] > width * 0.96:
            continue
        if bbox[3] - bbox[1] > height * 0.92 or bbox_area > width * height * 0.32:
            continue
        glyph_count = sum(item[1] for item in group)
        if (
            glyph_count <= 7
            and _local_primitive_count(
                stroke_primitives,
                bbox,
                width=width,
                height=height,
            )
            >= 3
        ):
            continue
        if _attached_to_large_visual_edge(
            stroke_primitives,
            stroke_labels,
            bbox,
        ):
            continue
        median_height = float(np.median([max(1, item[0][3] - item[0][1]) for item in group]))
        padding = max(1, int(round(median_height * 0.10)))
        expanded = _clamp_bbox(
            (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding),
            width=width,
            height=height,
        )
        if expanded is None:
            continue
        polygons = [_contrast_polygon(strokes, item[0]) for item in group]
        confidence = min(0.94, 0.48 + 0.035 * min(glyph_count, 12))
        candidates.append(
            GlyphCandidate(
                bbox_page=expanded,
                polygon_page=_polygon_hull(polygons, expanded),
                detector_source="glyph_scan",
                confidence=confidence,
            )
        )
    return sorted(
        candidates[:_MAX_GLYPH_CANDIDATES],
        key=lambda item: (item.bbox_page[1], item.bbox_page[0], item.bbox_page),
    )


def _should_merge(left: _Evidence, right: _Evidence) -> bool:
    intersection = _bbox_intersection(left.bbox_page, right.bbox_page)
    if intersection <= 0:
        return False
    if _bbox_iou(left.bbox_page, right.bbox_page) >= 0.45:
        return True
    left_area = max(1, _bbox_area(left.bbox_page))
    right_area = max(1, _bbox_area(right.bbox_page))
    containment = intersection / float(min(left_area, right_area))
    area_ratio = max(left_area, right_area) / float(min(left_area, right_area))
    return containment >= 0.82 and area_ratio <= 3.0


def _collapse_evidence(group: Sequence[_Evidence]) -> _Evidence:
    bbox = _bbox_union(item.bbox_page for item in group)
    rotation_candidates = sorted(
        (
            (item.confidence, item.rotation_source or "", item.rotation_deg)
            for item in group
            if item.rotation_deg is not None
        ),
        key=lambda value: (-value[0], value[1], float(value[2] or 0.0)),
    )
    rotation_source = rotation_candidates[0][1] or None if rotation_candidates else None
    rotation_deg = rotation_candidates[0][2] if rotation_candidates else None
    return _Evidence(
        bbox_page=bbox,
        polygon_page=_polygon_hull((item.polygon_page for item in group), bbox),
        detector_sources=tuple(sorted({source for item in group for source in item.detector_sources})),
        confidence=max(item.confidence for item in group),
        evidence_ids=tuple(sorted({value for item in group for value in item.evidence_ids if value})),
        script_evidence=tuple(
            sorted({value for item in group for value in item.script_evidence if value})
        ),
        rotation_deg=rotation_deg,
        rotation_source=rotation_source,
        support_only=all(item.support_only for item in group),
    )


def _dedupe_evidence(evidence: Sequence[_Evidence]) -> list[_Evidence]:
    if not evidence:
        return []
    parents = list(range(len(evidence)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    # `_should_merge` requires positive intersection.  A sweep line therefore
    # avoids the previous all-pairs path while preserving identical grouping.
    ordered = sorted(
        range(len(evidence)),
        key=lambda index: (
            evidence[index].bbox_page[0],
            evidence[index].bbox_page[1],
            evidence[index].bbox_page[2],
            evidence[index].bbox_page[3],
            evidence[index].detector_sources,
        ),
    )
    active: list[int] = []
    for right in ordered:
        right_bbox = evidence[right].bbox_page
        active = [
            left
            for left in active
            if evidence[left].bbox_page[2] > right_bbox[0]
        ]
        for left in active:
            left_bbox = evidence[left].bbox_page
            if left_bbox[3] <= right_bbox[1] or right_bbox[3] <= left_bbox[1]:
                continue
            if _should_merge(evidence[left], evidence[right]):
                union(left, right)
        active.append(right)

    groups: dict[int, list[_Evidence]] = {}
    for index, item in enumerate(evidence):
        groups.setdefault(find(index), []).append(item)
    return [_collapse_evidence(group) for group in groups.values()]


def _as_evidence(
    item: DetectorRegion | GlyphCandidate,
    *,
    width: int,
    height: int,
) -> _Evidence | None:
    bbox = _clamp_bbox(item.bbox_page, width=width, height=height)
    if bbox is None:
        return None
    source = str(item.detector_source or "unknown_detector")
    evidence_id = str(item.evidence_id) if item.evidence_id else ""
    return _Evidence(
        bbox_page=bbox,
        polygon_page=_normalise_polygon(
            item.polygon_page,
            bbox,
            width=width,
            height=height,
        ),
        detector_sources=(source,),
        confidence=max(0.0, min(1.0, float(item.confidence))),
        evidence_ids=(evidence_id,) if evidence_id else (),
        script_evidence=tuple(sorted({str(value) for value in item.script_evidence if value})),
        rotation_deg=(float(item.rotation_deg) if item.rotation_deg is not None else None),
        rotation_source=(str(item.rotation_source) if item.rotation_source else None),
        support_only=bool(getattr(item, "support_only", False)),
    )


def _detector_supports_glyph(detector: _Evidence, glyph: _Evidence) -> bool:
    intersection = _bbox_intersection(detector.bbox_page, glyph.bbox_page)
    if intersection / float(max(1, _bbox_area(glyph.bbox_page))) < 0.65:
        return False
    center_x = (glyph.bbox_page[0] + glyph.bbox_page[2]) / 2.0
    center_y = (glyph.bbox_page[1] + glyph.bbox_page[3]) / 2.0
    return (
        detector.bbox_page[0] <= center_x <= detector.bbox_page[2]
        and detector.bbox_page[1] <= center_y <= detector.bbox_page[3]
    )


def _attach_detector_support(
    detectors: Sequence[_Evidence],
    glyphs: Sequence[_Evidence],
) -> list[_Evidence]:
    supported_glyphs = list(glyphs)
    unsupported_detectors: list[_Evidence] = []
    for detector in detectors:
        matches = [
            index
            for index, glyph in enumerate(supported_glyphs)
            if _detector_supports_glyph(detector, glyph)
        ]
        if not matches:
            if not detector.support_only:
                unsupported_detectors.append(detector)
            continue
        detector_area = float(max(1, _bbox_area(detector.bbox_page)))
        matched_glyphs = [supported_glyphs[index] for index in matches]
        tight_single_match = (
            len(matches) == 1
            and detector_area
            / float(max(1, _bbox_area(matched_glyphs[0].bbox_page)))
            <= 3.0
        )
        single_match_width = (
            matched_glyphs[0].bbox_page[2] - matched_glyphs[0].bbox_page[0]
            if len(matches) == 1
            else 0
        )
        single_match_height = (
            matched_glyphs[0].bbox_page[3] - matched_glyphs[0].bbox_page[1]
            if len(matches) == 1
            else 0
        )
        independently_supported_single_line = (
            len(matches) == 1
            and matched_glyphs[0].confidence >= 0.55
            and _bbox_area(matched_glyphs[0].bbox_page) / detector_area >= 0.003
            and min(single_match_width, single_match_height) >= 8
        )
        glyph_union = _bbox_union(item.bbox_page for item in matched_glyphs)
        broad_container_coverage = (
            sum(_bbox_area(item.bbox_page) for item in matched_glyphs) / detector_area
            >= 0.015
            and (glyph_union[2] - glyph_union[0])
            / float(max(1, detector.bbox_page[2] - detector.bbox_page[0]))
            >= 0.10
            and (glyph_union[3] - glyph_union[1])
            / float(max(1, detector.bbox_page[3] - detector.bbox_page[1]))
            >= 0.05
        )
        if (
            not tight_single_match
            and not independently_supported_single_line
            and not broad_container_coverage
        ):
            if not detector.support_only:
                unsupported_detectors.append(detector)
            continue
        for index in matches:
            glyph = supported_glyphs[index]
            supported_glyphs[index] = replace(
                glyph,
                detector_sources=tuple(
                    sorted({*glyph.detector_sources, *detector.detector_sources})
                ),
                confidence=max(glyph.confidence, detector.confidence),
                evidence_ids=tuple(sorted({*glyph.evidence_ids, *detector.evidence_ids})),
                script_evidence=tuple(
                    sorted({*glyph.script_evidence, *detector.script_evidence})
                ),
                rotation_deg=(
                    glyph.rotation_deg
                    if glyph.rotation_deg is not None
                    else detector.rotation_deg
                ),
                rotation_source=(
                    glyph.rotation_source
                    if glyph.rotation_deg is not None
                    else detector.rotation_source
                ),
            )
    return [*supported_glyphs, *unsupported_detectors]


def discover_source_text_components(
    page_rgb: np.ndarray,
    *,
    page_id: str,
    detector_regions: Sequence[DetectorRegion],
    glyph_candidates: Sequence[GlyphCandidate] = (),
) -> list[SourceTextComponent]:
    """Discover where source text exists without consulting recognized payloads."""

    if not isinstance(page_rgb, np.ndarray) or page_rgb.size == 0:
        return []
    height, width = page_rgb.shape[:2]
    all_glyph_candidates = [*glyph_candidates, *scan_textlike_glyph_candidates(page_rgb)]
    detectors = [
        evidence
        for item in detector_regions
        if (evidence := _as_evidence(item, width=width, height=height)) is not None
    ]
    glyphs = [
        evidence
        for item in all_glyph_candidates
        if (evidence := _as_evidence(item, width=width, height=height)) is not None
    ]
    glyphs = _dedupe_evidence(glyphs)
    evidence = _attach_detector_support(detectors, glyphs)
    grouped = _dedupe_evidence(evidence)
    grouped.sort(
        key=lambda item: (
            item.bbox_page[1],
            item.bbox_page[0],
            item.bbox_page[3],
            item.bbox_page[2],
            item.detector_sources,
        )
    )

    seeds = [
        ComponentSeed(
            bbox_page=item.bbox_page,
            detector_source="+".join(item.detector_sources),
        )
        for item in grouped
    ]
    component_ids = assign_component_ids(page_id, seeds)
    return [
        SourceTextComponent(
            component_id=component_id,
            page_id=page_id,
            bbox_page=item.bbox_page,
            polygon_page=item.polygon_page,
            detector_sources=item.detector_sources,
            confidence=item.confidence,
            script_evidence=item.script_evidence,
            evidence_ids=item.evidence_ids,
            rotation_deg=item.rotation_deg,
            rotation_source=item.rotation_source,
        )
        for item, component_id in zip(grouped, component_ids)
    ]
