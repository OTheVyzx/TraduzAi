from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

DEFAULT_BALLOON_FONT = "ComicNeue-Bold.ttf"
IMPACT_FONT_CHOICES = ("KOMIKAX_.ttf", "LuckiestGuy-Regular.ttf")
VISUAL_CARD_FONT_CHOICES = ("LeagueGothic-Regular-VariableFont_wdth.ttf",)
ENABLE_CURVE_DETECTION = False


@dataclass(frozen=True)
class TextStyleEvidence:
    source: str
    text_color: str
    text_color_confidence: float
    stroke_color: str
    stroke_width_px: int
    stroke_confidence: float
    shadow: bool
    shadow_confidence: float
    glow: bool
    glow_confidence: float
    font_name: str
    font_confidence: float
    shadow_color: str = ""
    shadow_offset: list[int] | None = None
    glow_color: str = ""
    glow_px: int = 0
    gradient: bool = False
    gradient_colors: list[str] | None = None
    gradient_confidence: float = 0.0
    curved: bool = False
    curve_direction: str = ""
    curve_amount: float = 0.0
    curve_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def extract_text_style_evidence(
    crop_rgb: np.ndarray,
    font_detector: object | None = None,
    font_context: str | None = None,
) -> TextStyleEvidence:
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.ndim < 3:
        return _empty()

    rgb_crop = crop_rgb[:, :, :3]
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    def with_font(evidence: TextStyleEvidence) -> TextStyleEvidence:
        return _with_font_evidence(evidence, rgb_crop, font_detector, font_context=font_context)

    if _looks_like_non_text_texture(rgb_crop):
        return with_font(_empty())
    if contrast < 20.0:
        sparse_evidence = _detect_sparse_foreground_evidence(rgb_crop)
        if sparse_evidence is not None:
            return with_font(_with_effect_evidence(sparse_evidence, rgb_crop, gray, contrast))
        return with_font(_empty())

    dark_threshold = float(np.percentile(gray, 5) + max(8.0, contrast * 0.1))
    dark_mask = gray <= dark_threshold
    if int(np.count_nonzero(dark_mask)) < 8:
        light_text_evidence = _detect_light_text_on_dark_background(rgb_crop, gray, contrast)
        if light_text_evidence is not None:
            return with_font(_with_effect_evidence(light_text_evidence, rgb_crop, gray, contrast))
        return with_font(_empty())

    light_fill_colored_background = _detect_light_fill_on_colored_background(
        rgb_crop,
        gray,
        dark_mask,
        contrast,
    )
    if light_fill_colored_background is not None:
        return with_font(_with_effect_evidence(light_fill_colored_background, rgb_crop, gray, contrast))

    colored_fill_light_outline = _detect_colored_fill_light_outline(rgb_crop, gray, contrast)
    if colored_fill_light_outline is not None:
        return with_font(_with_effect_evidence(colored_fill_light_outline, rgb_crop, gray, contrast))

    light_fill_dark_outline = _detect_light_fill_dark_outline(rgb_crop, gray, dark_mask, contrast)
    dark_fill_light_outline = _detect_dark_fill_light_outline(rgb_crop, gray, dark_mask, contrast)
    outline_evidence = _choose_outline_evidence(
        rgb_crop,
        light_fill_dark_outline,
        dark_fill_light_outline,
    )
    if outline_evidence is not None:
        return with_font(_with_effect_evidence(outline_evidence, rgb_crop, gray, contrast))

    light_text_evidence = _detect_light_text_on_dark_background(rgb_crop, gray, contrast)
    if light_text_evidence is not None:
        return with_font(_with_effect_evidence(light_text_evidence, rgb_crop, gray, contrast))

    text_pixels = rgb_crop[dark_mask]
    text_color = tuple(int(round(float(v))) for v in np.median(text_pixels, axis=0))
    confidence = min(1.0, max(0.0, contrast / 96.0))

    evidence = TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(text_color),
        text_color_confidence=confidence,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )
    return with_font(_with_effect_evidence(evidence, rgb_crop, gray, contrast))


def extract_sfx_style_evidence(
    crop_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    layer: dict | None = None,
    font_detector: object | None = None,
):
    from sfx.style import extract_manhwa_sfx_style

    return extract_manhwa_sfx_style(
        crop_rgb,
        mask,
        layer=layer,
        font_detector=font_detector,
    )


def _with_font_evidence(
    evidence: TextStyleEvidence,
    crop_rgb: np.ndarray,
    font_detector: object | None,
    *,
    font_context: str | None = None,
) -> TextStyleEvidence:
    allowed_fonts = _font_choices_for_context(font_context)
    if str(font_context or "").strip().lower() == "visual_card" and font_detector is not None:
        return _apply_detector_font_evidence(evidence, crop_rgb, font_detector, allowed_fonts)

    if not _is_heavy_text_crop(crop_rgb):
        evidence = _clear_plain_balloon_false_outline(evidence, crop_rgb)
        return replace(evidence, font_name=DEFAULT_BALLOON_FONT, font_confidence=1.0)

    if font_detector is None:
        return replace(
            evidence,
            font_name=allowed_fonts[0],
            font_confidence=0.5,
        )

    return _apply_detector_font_evidence(evidence, crop_rgb, font_detector, allowed_fonts)


def _font_choices_for_context(font_context: str | None) -> tuple[str, ...]:
    if str(font_context or "").strip().lower() == "visual_card":
        return VISUAL_CARD_FONT_CHOICES
    return IMPACT_FONT_CHOICES


def _apply_detector_font_evidence(
    evidence: TextStyleEvidence,
    crop_rgb: np.ndarray,
    font_detector: object,
    allowed_fonts: tuple[str, ...],
) -> TextStyleEvidence:
    try:
        if hasattr(font_detector, "detect_with_score"):
            font_name, font_confidence = font_detector.detect_with_score(
                crop_rgb,
                allow_default=False,
            )
        else:
            font_name = font_detector.detect(crop_rgb, allow_default=False)
            font_confidence = 0.5
    except Exception:
        return evidence

    if not isinstance(font_name, str) or not font_name:
        return evidence

    if font_name not in allowed_fonts:
        font_name = allowed_fonts[0]

    try:
        confidence = float(font_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    return replace(evidence, font_name=font_name, font_confidence=confidence)


def _clear_plain_balloon_false_outline(
    evidence: TextStyleEvidence,
    crop_rgb: np.ndarray,
) -> TextStyleEvidence:
    if not evidence.stroke_color:
        return evidence
    if evidence.shadow or evidence.glow:
        return evidence
    if not _is_light_hex(evidence.text_color) or not _is_dark_hex(evidence.stroke_color):
        return evidence

    background = _estimate_border_color(crop_rgb[:, :, :3])
    background_luma = _luma(background)
    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    if background_luma < 220.0:
        return evidence
    if float(np.linalg.norm(text_rgb - background.astype(np.float32))) > 35.0:
        return evidence

    return replace(
        evidence,
        text_color="#000000",
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
    )


def _is_heavy_text_crop(crop_rgb: np.ndarray) -> bool:
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.ndim < 3:
        return False

    rgb_crop = crop_rgb[:, :, :3]
    height, width = rgb_crop.shape[:2]
    if height < 8 or width < 8:
        return False

    border = np.concatenate(
        [
            rgb_crop[:2].reshape(-1, 3),
            rgb_crop[-2:].reshape(-1, 3),
            rgb_crop[:, :2].reshape(-1, 3),
            rgb_crop[:, -2:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(
        rgb_crop.astype(np.float32) - background.astype(np.float32),
        axis=2,
    )
    foreground = distance > 35.0
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 8:
        return False

    ys, xs = np.where(foreground)
    bbox_area = max(1, int(xs.max() - xs.min() + 1) * int(ys.max() - ys.min() + 1))
    foreground_ratio = foreground_count / float(height * width)
    bbox_fill_ratio = foreground_count / float(bbox_area)
    ink_height = int(ys.max() - ys.min() + 1)
    ink_width = int(xs.max() - xs.min() + 1)
    if foreground_ratio >= 0.16 and bbox_fill_ratio >= 0.62 and ink_height >= 42:
        return True
    if bbox_fill_ratio >= 0.68 and ink_height >= 40 and ink_width >= 120:
        return True
    return foreground_ratio >= 0.35 and bbox_fill_ratio >= 0.50


def _looks_like_non_text_texture(crop_rgb: np.ndarray) -> bool:
    if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.ndim < 3:
        return False

    rgb_crop = crop_rgb[:, :, :3]
    height, width = rgb_crop.shape[:2]
    if height < 32 or width < 32:
        return False
    if _is_heavy_text_crop(rgb_crop):
        return False

    background = _estimate_border_color(rgb_crop).astype(np.float32)
    distance = np.linalg.norm(rgb_crop.astype(np.float32) - background, axis=2)
    foreground = distance > 35.0
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 80:
        return False

    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8),
        8,
    )
    components: list[tuple[int, int, int, float, float]] = []
    for label in range(1, labels_count):
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 20 or comp_w <= 0 or comp_h <= 0:
            continue
        fill_ratio = area / float(max(1, comp_w * comp_h))
        aspect = comp_w / float(max(1, comp_h))
        components.append((area, comp_w, comp_h, fill_ratio, aspect))
    if not components:
        return False

    large = [comp for comp in components if comp[0] >= 80]
    if len(large) == 1:
        area, comp_w, comp_h, fill_ratio, aspect = large[0]
        if comp_h >= height * 0.70 and comp_w <= width * 0.18 and aspect <= 0.80:
            return True

    low_density_tall = [
        comp
        for comp in large
        if comp[2] >= height * 0.55 and comp[3] <= 0.24 and comp[4] <= 0.80
    ]
    if len(low_density_tall) >= 3:
        return True

    horizontal_bars = [
        comp
        for comp in large
        if comp[2] <= max(16, height * 0.16) and comp[4] >= 3.5 and comp[3] >= 0.80
    ]
    if len(horizontal_bars) >= 5:
        return True

    return False


def _detect_sparse_foreground_evidence(crop_rgb: np.ndarray) -> TextStyleEvidence | None:
    rgb_crop = crop_rgb[:, :, :3]
    background = _estimate_border_color(rgb_crop).astype(np.float32)
    distance = np.linalg.norm(rgb_crop.astype(np.float32) - background, axis=2)
    foreground = distance > 35.0
    foreground_count = int(np.count_nonzero(foreground))
    if foreground_count < 12 or foreground_count > rgb_crop.shape[0] * rgb_crop.shape[1] * 0.35:
        return None

    ys, xs = np.where(foreground)
    if int(xs.max() - xs.min() + 1) < 16 or int(ys.max() - ys.min() + 1) < 8:
        return None

    text_pixels = rgb_crop[foreground]
    text_color = tuple(int(round(float(v))) for v in np.median(text_pixels, axis=0))
    color_distance = float(np.linalg.norm(np.array(text_color, dtype=np.float32) - background))
    if color_distance < 35.0:
        return None

    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(text_color),
        text_color_confidence=min(1.0, max(0.5, color_distance / 180.0)),
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _choose_outline_evidence(
    rgb_crop: np.ndarray,
    light_fill_dark_outline: TextStyleEvidence | None,
    dark_fill_light_outline: TextStyleEvidence | None,
) -> TextStyleEvidence | None:
    if light_fill_dark_outline is None:
        return dark_fill_light_outline
    if dark_fill_light_outline is None:
        return light_fill_dark_outline

    light_score = _outline_role_score(rgb_crop, light_fill_dark_outline)
    dark_score = _outline_role_score(rgb_crop, dark_fill_light_outline)
    if light_score + 0.10 <= dark_score:
        return light_fill_dark_outline
    if dark_score + 0.10 <= light_score:
        return dark_fill_light_outline

    background = _estimate_border_color(rgb_crop).astype(np.float32)
    light_stroke = np.array(_parse_hex_color(light_fill_dark_outline.stroke_color), dtype=np.float32)
    dark_stroke = np.array(_parse_hex_color(dark_fill_light_outline.stroke_color), dtype=np.float32)
    if float(np.linalg.norm(light_stroke - background)) < float(np.linalg.norm(dark_stroke - background)):
        return dark_fill_light_outline
    return light_fill_dark_outline


def _outline_role_score(rgb_crop: np.ndarray, evidence: TextStyleEvidence) -> float:
    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    stroke_rgb = np.array(_parse_hex_color(evidence.stroke_color), dtype=np.float32)
    dist_text = np.linalg.norm(rgb_crop.astype(np.float32) - text_rgb, axis=2)
    dist_stroke = np.linalg.norm(rgb_crop.astype(np.float32) - stroke_rgb, axis=2)
    fill_mask = dist_text <= 42.0
    stroke_mask = (dist_stroke <= 42.0) & ~fill_mask
    fill_area = _median_component_area(fill_mask)
    stroke_area = _median_component_area(stroke_mask)
    if fill_area <= 0.0 or stroke_area <= 0.0:
        return 0.0
    return min(4.0, stroke_area / fill_area)


def _median_component_area(mask: np.ndarray) -> float:
    labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas: list[int] = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 12:
            areas.append(area)
    if not areas:
        return 0.0
    return float(np.median(np.array(areas, dtype=np.float32)))


def _detect_colored_fill_light_outline(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    contrast: float,
) -> TextStyleEvidence | None:
    if contrast < 70.0:
        return None

    background = _estimate_border_color(rgb_crop).astype(np.float32)
    distance_to_bg = np.linalg.norm(rgb_crop.astype(np.float32) - background, axis=2)
    foreground = distance_to_bg > 45.0
    if int(np.count_nonzero(foreground)) < 24:
        return None

    light_mask = foreground & (gray >= 220)
    colored_mask = foreground & (gray < 220)
    if int(np.count_nonzero(light_mask)) < 24 or int(np.count_nonzero(colored_mask)) < 24:
        return None

    colored_pixels = rgb_crop[colored_mask]
    if float(np.median([_saturation(px) for px in colored_pixels[:: max(1, len(colored_pixels) // 256)]])) < 0.18:
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    colored_near = cv2.dilate(colored_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    adjacency = int(np.count_nonzero(colored_near & light_mask))
    if adjacency < max(12, int(np.count_nonzero(colored_mask) * 0.08)):
        return None

    fill_rgb = tuple(int(round(float(v))) for v in np.median(colored_pixels, axis=0))
    stroke_rgb = tuple(int(round(float(v))) for v in np.median(rgb_crop[light_mask], axis=0))
    if _luma(stroke_rgb) < 210.0:
        return None

    distances = cv2.distanceTransform(light_mask.astype(np.uint8), cv2.DIST_L2, 3)
    nonzero = distances[distances > 0]
    stroke_width = int(round(float(np.percentile(nonzero, 90)))) if nonzero.size else 1
    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(fill_rgb),
        text_color_confidence=min(1.0, max(0.62, contrast / 255.0)),
        stroke_color=_hex_color(stroke_rgb),
        stroke_width_px=max(1, min(8, stroke_width)),
        stroke_confidence=0.72,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _detect_light_fill_dark_outline(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    dark_mask: np.ndarray,
    contrast: float,
) -> TextStyleEvidence | None:
    if contrast < 80.0:
        return None

    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return None

    light_threshold = float(np.percentile(gray, 95) - max(8.0, contrast * 0.1))
    light_mask = gray >= light_threshold
    if int(np.count_nonzero(light_mask)) < 8:
        return None

    dark_count, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8),
        8,
    )
    if dark_count <= 1:
        return None

    light_count, light_labels, light_stats, _ = cv2.connectedComponentsWithStats(
        light_mask.astype(np.uint8),
        8,
    )
    if light_count <= 1:
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_dark = cv2.dilate(dark_mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    best: tuple[float, int, int] | None = None
    for light_label in range(1, light_count):
        x, y, component_width, component_height, area = light_stats[light_label]
        if area < 8:
            continue
        if x <= 0 or y <= 0 or x + component_width >= width or y + component_height >= height:
            continue

        fill_mask = light_labels == light_label
        adjacency = int(np.count_nonzero(fill_mask & dilated_dark))
        adjacency_ratio = adjacency / float(area)
        if adjacency_ratio < 0.03:
            continue

        fill_dilated = cv2.dilate(fill_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        touching_dark_labels = dark_labels[fill_dilated & dark_mask]
        touching_dark_labels = touching_dark_labels[touching_dark_labels > 0]
        if touching_dark_labels.size == 0:
            continue

        dark_label = int(np.bincount(touching_dark_labels).argmax())
        dark_x, dark_y, dark_width, dark_height, dark_area = dark_stats[dark_label]
        if dark_area < 8:
            continue
        if (
            dark_x > x
            or dark_y > y
            or dark_x + dark_width < x + component_width
            or dark_y + dark_height < y + component_height
        ):
            continue

        score = adjacency_ratio * min(1.0, float(area) / max(1.0, float(dark_area)))
        if best is None or score > best[0]:
            best = (score, light_label, dark_label)

    if best is None:
        return None

    _, light_label, dark_label = best
    fill_mask = light_labels == light_label
    stroke_mask = dark_labels == dark_label
    fill_pixels = rgb_crop[fill_mask]
    stroke_pixels = rgb_crop[stroke_mask]
    if fill_pixels.size == 0 or stroke_pixels.size == 0:
        return None

    fill_rgb = tuple(int(round(float(v))) for v in np.median(fill_pixels, axis=0))
    stroke_rgb = tuple(int(round(float(v))) for v in np.median(stroke_pixels, axis=0))
    if _color_close_to_background(fill_rgb, rgb_crop, tolerance=55.0):
        return None
    fill_luma = float(np.median(gray[fill_mask]))
    stroke_luma = float(np.median(gray[stroke_mask]))
    luma_delta = abs(fill_luma - stroke_luma)
    if luma_delta < 80.0:
        return None

    stroke_distances = cv2.distanceTransform(
        stroke_mask.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    nonzero_distances = stroke_distances[stroke_distances > 0]
    if nonzero_distances.size == 0:
        return None

    stroke_width = int(round(float(np.percentile(nonzero_distances, 95))))
    stroke_width = max(1, min(8, stroke_width))
    confidence = min(1.0, max(0.5, luma_delta / 255.0))

    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(fill_rgb),
        text_color_confidence=confidence,
        stroke_color=_hex_color(stroke_rgb),
        stroke_width_px=stroke_width,
        stroke_confidence=confidence,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _detect_dark_fill_light_outline(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    dark_mask: np.ndarray,
    contrast: float,
) -> TextStyleEvidence | None:
    if contrast < 70.0:
        return None

    height, width = gray.shape[:2]
    if height < 6 or width < 6:
        return None

    background = _estimate_border_color(rgb_crop)
    color_distance = np.linalg.norm(
        rgb_crop.astype(np.float32) - background.astype(np.float32),
        axis=2,
    )
    foreground = color_distance > 35.0
    light_threshold = float(np.percentile(gray, 92))
    light_mask = foreground & (gray >= light_threshold)
    if int(np.count_nonzero(light_mask)) < 8:
        return None

    dark_count, dark_labels, dark_stats, _ = cv2.connectedComponentsWithStats(
        dark_mask.astype(np.uint8),
        8,
    )
    light_count, light_labels, light_stats, _ = cv2.connectedComponentsWithStats(
        light_mask.astype(np.uint8),
        8,
    )
    if dark_count <= 1 or light_count <= 1:
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    best: tuple[float, int, int] | None = None
    for dark_label in range(1, dark_count):
        x, y, component_width, component_height, dark_area = dark_stats[dark_label]
        if dark_area < 8:
            continue
        if x <= 0 or y <= 0 or x + component_width >= width or y + component_height >= height:
            continue
        fill_mask = dark_labels == dark_label
        fill_dilated = cv2.dilate(fill_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
        touching_light = light_labels[fill_dilated & light_mask]
        touching_light = touching_light[touching_light > 0]
        if touching_light.size == 0:
            continue

        light_label = int(np.bincount(touching_light).argmax())
        light_x, light_y, light_width, light_height, light_area = light_stats[light_label]
        if light_area < 8:
            continue
        if (
            light_x > x
            or light_y > y
            or light_x + light_width < x + component_width
            or light_y + light_height < y + component_height
        ):
            continue

        stroke_mask = light_labels == light_label
        adjacency = int(np.count_nonzero(fill_dilated & stroke_mask))
        score = adjacency / float(max(1, dark_area))
        if best is None or score > best[0]:
            best = (score, dark_label, light_label)

    if best is None:
        return None

    _, dark_label, light_label = best
    fill_mask = dark_labels == dark_label
    stroke_mask = light_labels == light_label
    fill_pixels = rgb_crop[fill_mask]
    stroke_pixels = rgb_crop[stroke_mask]
    if fill_pixels.size == 0 or stroke_pixels.size == 0:
        return None

    fill_luma = float(np.median(gray[fill_mask]))
    stroke_luma = float(np.median(gray[stroke_mask]))
    if stroke_luma - fill_luma < 80.0:
        return None

    stroke_distances = cv2.distanceTransform(
        stroke_mask.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    nonzero_distances = stroke_distances[stroke_distances > 0]
    stroke_width = int(round(float(np.percentile(nonzero_distances, 95)))) if nonzero_distances.size else 1
    stroke_width = max(1, min(8, stroke_width))
    confidence = min(1.0, max(0.5, (stroke_luma - fill_luma) / 255.0))

    fill_rgb = tuple(int(round(float(v))) for v in np.median(fill_pixels, axis=0))
    stroke_rgb = tuple(int(round(float(v))) for v in np.median(stroke_pixels, axis=0))
    background_tolerance = 95.0 if _saturation(_estimate_border_color(rgb_crop[:, :, :3])) >= 0.25 else 55.0
    if _color_close_to_background(fill_rgb, rgb_crop, tolerance=background_tolerance):
        return None
    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(fill_rgb),
        text_color_confidence=confidence,
        stroke_color=_hex_color(stroke_rgb),
        stroke_width_px=stroke_width,
        stroke_confidence=confidence,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _detect_light_fill_on_colored_background(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    dark_mask: np.ndarray,
    contrast: float,
) -> TextStyleEvidence | None:
    if contrast < 60.0:
        return None

    height, width = gray.shape[:2]
    if height < 8 or width < 8:
        return None

    background = _estimate_border_color(rgb_crop[:, :, :3])
    if _saturation(background) < 0.20:
        return None

    light_threshold = float(np.percentile(gray, 95) - max(6.0, contrast * 0.08))
    light_mask = gray >= light_threshold
    light_count = int(np.count_nonzero(light_mask))
    if light_count < 16 or light_count > gray.size * 0.45:
        return None

    background_distance = np.linalg.norm(
        rgb_crop.astype(np.float32) - background.astype(np.float32),
        axis=2,
    )
    foreground = background_distance > 35.0
    non_light_foreground = foreground & ~light_mask
    border_guard = np.zeros(gray.shape, dtype=bool)
    border_guard[:2, :] = True
    border_guard[-2:, :] = True
    border_guard[:, :2] = True
    border_guard[:, -2:] = True
    interior_non_light = non_light_foreground & ~border_guard
    dark_interior = interior_non_light & (gray <= 95.0)
    if int(np.count_nonzero(dark_interior)) > light_count * 0.25:
        return None
    interior_limit = light_count * (2.0 if _saturation(background) >= 0.25 else 0.35)
    if int(np.count_nonzero(interior_non_light)) > interior_limit:
        return None

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        light_mask.astype(np.uint8),
        8,
    )
    if component_count <= 1:
        return None

    fill_mask = np.zeros(gray.shape, dtype=bool)
    total_fill_area = 0
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < 8:
            continue
        if x <= 0 or y <= 0 or x + component_width >= width or y + component_height >= height:
            continue
        total_fill_area += int(area)
        fill_mask |= labels == label

    if total_fill_area < 16:
        return None

    fill_pixels = rgb_crop[fill_mask]
    fill_rgb = tuple(int(round(float(v))) for v in np.median(fill_pixels, axis=0))
    if _luma(fill_rgb) < 210.0:
        return None
    fill_luma_spread = float(np.percentile(gray[fill_mask], 90) - np.percentile(gray[fill_mask], 10))
    fill_color_spread = float(np.mean(np.std(fill_pixels.astype(np.float32), axis=0)))
    if fill_luma_spread > 34.0 or fill_color_spread > 30.0:
        return None

    kernel = np.ones((3, 3), dtype=np.uint8)
    ring = cv2.dilate(fill_mask.astype(np.uint8), kernel, iterations=2).astype(bool) & ~fill_mask
    possible_outline = ring & dark_mask
    stroke_color = ""
    stroke_width = 0
    stroke_confidence = 0.0
    if int(np.count_nonzero(possible_outline)) >= 12:
        stroke_pixels = rgb_crop[possible_outline]
        stroke_rgb = tuple(int(round(float(v))) for v in np.median(stroke_pixels, axis=0))
        if _luma(stroke_rgb) <= 80.0:
            stroke_color = _hex_color(stroke_rgb)
            stroke_width = 1
            stroke_confidence = 0.55

    fill_luma = float(np.median(gray[fill_mask]))
    background_luma = _luma(background)
    confidence = min(1.0, max(0.55, abs(fill_luma - background_luma) / 180.0))

    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(fill_rgb),
        text_color_confidence=confidence,
        stroke_color=stroke_color,
        stroke_width_px=stroke_width,
        stroke_confidence=stroke_confidence,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _detect_light_text_on_dark_background(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    contrast: float,
) -> TextStyleEvidence | None:
    if contrast < 70.0:
        return None

    background_luma = float(np.percentile(gray, 20))
    if background_luma > 80.0:
        return None

    light_threshold = float(np.percentile(gray, 95) - max(8.0, contrast * 0.1))
    light_mask = gray >= light_threshold
    light_count = int(np.count_nonzero(light_mask))
    total = int(gray.size)
    if light_count < 8 or light_count > total * 0.35:
        return None

    light_pixels = rgb_crop[light_mask]
    text_color = tuple(int(round(float(v))) for v in np.median(light_pixels, axis=0))
    text_luma = float(np.median(gray[light_mask]))
    if text_luma - background_luma < 80.0:
        return None

    confidence = min(1.0, max(0.5, (text_luma - background_luma) / 180.0))
    return TextStyleEvidence(
        source="pixel_analysis",
        text_color=_hex_color(text_color),
        text_color_confidence=confidence,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )


def _with_effect_evidence(
    evidence: TextStyleEvidence,
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    contrast: float,
) -> TextStyleEvidence:
    outline_details = _detect_adjacent_dark_outline_details(rgb_crop, gray, evidence)
    if outline_details["confidence"] >= 0.55:
        evidence = replace(
            evidence,
            stroke_color=outline_details["color"],
            stroke_width_px=outline_details["px"],
            stroke_confidence=outline_details["confidence"],
        )
    shadow_details = _detect_shadow_details(rgb_crop, gray, contrast)
    glow_details = _detect_glow_details(rgb_crop, gray, contrast)
    light_card_glow_details = _detect_light_glow_on_colored_background(rgb_crop, gray, evidence)
    if light_card_glow_details["confidence"] > glow_details["confidence"]:
        glow_details = light_card_glow_details
    if _colored_background_stroke_is_glow(rgb_crop, evidence):
        glow_color = evidence.text_color if _is_light_hex(evidence.text_color) else evidence.stroke_color
        glow_details = {
            "confidence": 0.92,
            "color": glow_color,
            "px": max(2, min(6, int(evidence.stroke_width_px))),
        }
        evidence = replace(
            evidence,
            stroke_color="",
            stroke_width_px=0,
            stroke_confidence=0.0,
        )
    if _colored_fill_dark_stroke_is_dark_text_glow(rgb_crop, evidence):
        glow_details = {
            "confidence": 0.92,
            "color": evidence.text_color,
            "px": max(2, min(6, int(evidence.stroke_width_px))),
        }
        evidence = replace(
            evidence,
            text_color=evidence.stroke_color,
            stroke_color="",
            stroke_width_px=0,
            stroke_confidence=0.0,
        )
    if _colored_fill_light_stroke_is_light_text_glow(rgb_crop, evidence):
        glow_color = evidence.stroke_color
        evidence = replace(
            evidence,
            text_color=evidence.stroke_color,
            stroke_color="",
            stroke_width_px=0,
            stroke_confidence=0.0,
        )
        glow_details = {
            "confidence": max(float(glow_details.get("confidence") or 0.0), 0.90),
            "color": glow_color,
            "px": max(2, min(5, int(evidence.stroke_width_px or 2))),
        }
    if _stroke_matches_background(rgb_crop, evidence):
        evidence = replace(
            evidence,
            stroke_color="",
            stroke_width_px=0,
            stroke_confidence=0.0,
        )
    if _solid_stroke_absorbs_glow(evidence, glow_details):
        glow_details = {"confidence": 0.0, "color": "", "px": 0}
    gradient_details = _detect_gradient_details(rgb_crop, gray, evidence, contrast)
    curve_details = (
        _detect_curve_details(rgb_crop, gray, evidence)
        if ENABLE_CURVE_DETECTION
        else {"confidence": 0.0, "direction": "", "amount": 0.0}
    )
    shadow_confidence = shadow_details["confidence"]
    glow_confidence = glow_details["confidence"]

    shadow_detected = shadow_confidence >= 0.9
    glow_detected = glow_confidence >= 0.88
    if shadow_detected and glow_detected:
        if shadow_confidence >= glow_confidence + 0.12:
            glow_detected = False
            glow_confidence = 0.0
        elif glow_confidence >= shadow_confidence + 0.12:
            shadow_detected = False
            shadow_confidence = 0.0
        else:
            shadow_detected = False
            glow_detected = False
            shadow_confidence = 0.0
            glow_confidence = 0.0

    return replace(
        evidence,
        shadow=shadow_detected,
        shadow_confidence=shadow_confidence if shadow_detected else 0.0,
        shadow_color=shadow_details["color"] if shadow_detected else "",
        shadow_offset=shadow_details["offset"] if shadow_detected else None,
        glow=glow_detected,
        glow_confidence=glow_confidence if glow_detected else 0.0,
        glow_color=glow_details["color"] if glow_detected else "",
        glow_px=glow_details["px"] if glow_detected else 0,
        gradient=gradient_details["confidence"] >= 0.60,
        gradient_colors=gradient_details["colors"] if gradient_details["confidence"] >= 0.60 else None,
        gradient_confidence=gradient_details["confidence"] if gradient_details["confidence"] >= 0.60 else 0.0,
        curved=curve_details["confidence"] >= 0.60,
        curve_direction=curve_details["direction"] if curve_details["confidence"] >= 0.60 else "",
        curve_amount=curve_details["amount"] if curve_details["confidence"] >= 0.60 else 0.0,
        curve_confidence=curve_details["confidence"] if curve_details["confidence"] >= 0.60 else 0.0,
    )


def _solid_stroke_absorbs_glow(evidence: TextStyleEvidence, glow_details: dict) -> bool:
    try:
        glow_confidence = float(glow_details.get("confidence") or 0.0)
        glow_px = int(round(float(glow_details.get("px") or 0)))
    except (TypeError, ValueError):
        return False
    if glow_confidence < 0.6 or glow_px <= 0:
        return False
    if not evidence.stroke_color or evidence.stroke_confidence < 0.5 or evidence.stroke_width_px <= 0:
        return False
    if not _colors_close_hex(str(glow_details.get("color") or ""), evidence.stroke_color, tolerance=36.0):
        return False

    # A solid contour is a compact band glued to the glyph. A glow should extend
    # beyond that band; otherwise the same pixels are just being double-counted.
    return glow_px <= int(evidence.stroke_width_px) + 2


def _colored_background_stroke_is_glow(rgb_crop: np.ndarray, evidence: TextStyleEvidence) -> bool:
    if not evidence.text_color:
        return False
    if not evidence.stroke_color or evidence.stroke_width_px < 3:
        return False
    stroke_rgb = np.array(_parse_hex_color(evidence.stroke_color), dtype=np.float32)
    if _is_dark_hex(evidence.stroke_color) or _is_light_hex(evidence.stroke_color):
        return False
    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    if _saturation(background) < 0.25 or _saturation(stroke_rgb) < 0.25:
        return False
    if not (_is_light_hex(evidence.text_color) or _is_dark_hex(evidence.text_color)):
        return False
    return float(np.linalg.norm(stroke_rgb - background)) <= 85.0


def _colored_fill_dark_stroke_is_dark_text_glow(rgb_crop: np.ndarray, evidence: TextStyleEvidence) -> bool:
    if not evidence.text_color or not evidence.stroke_color:
        return False
    if not _is_dark_hex(evidence.stroke_color) or evidence.stroke_width_px < 2:
        return False
    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    if _saturation(text_rgb) < 0.20:
        return False
    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    if _saturation(background) < 0.20:
        return False
    return float(np.linalg.norm(text_rgb - background)) <= 110.0


def _colored_fill_light_stroke_is_light_text_glow(rgb_crop: np.ndarray, evidence: TextStyleEvidence) -> bool:
    if not evidence.text_color or not evidence.stroke_color:
        return False
    if not _is_light_hex(evidence.stroke_color) or evidence.stroke_width_px < 1:
        return False
    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    if _saturation(text_rgb) < 0.22:
        return False
    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    if _saturation(background) < 0.22:
        return False
    if float(np.linalg.norm(text_rgb - background)) > 95.0:
        return False
    stroke_rgb = np.array(_parse_hex_color(evidence.stroke_color), dtype=np.float32)
    return _luma(stroke_rgb) >= _luma(text_rgb) + 35.0


def _stroke_matches_background(rgb_crop: np.ndarray, evidence: TextStyleEvidence) -> bool:
    if not evidence.stroke_color or not evidence.text_color:
        return False
    stroke_rgb = np.array(_parse_hex_color(evidence.stroke_color), dtype=np.float32)
    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    if float(np.linalg.norm(stroke_rgb - background)) > 30.0:
        return False
    return evidence.stroke_confidence <= 1.0 and evidence.stroke_width_px <= 3


def _detect_adjacent_dark_outline_details(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    evidence: TextStyleEvidence,
) -> dict:
    if not evidence.text_color or not _is_light_hex(evidence.text_color):
        return {"confidence": 0.0, "color": "", "px": 0}

    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    text_distance = np.linalg.norm(rgb_crop.astype(np.float32) - text_rgb, axis=2)
    text_mask = text_distance <= 55.0
    if int(np.count_nonzero(text_mask)) < 16:
        return {"confidence": 0.0, "color": "", "px": 0}

    kernel = np.ones((3, 3), dtype=np.uint8)
    near = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=2).astype(bool) & ~text_mask
    if int(np.count_nonzero(near)) < 16:
        return {"confidence": 0.0, "color": "", "px": 0}

    dark_threshold = min(95.0, float(np.percentile(gray, 18)) + 10.0)
    dark_ring = near & (gray <= dark_threshold)
    dark_count = int(np.count_nonzero(dark_ring))
    near_count = int(np.count_nonzero(near))
    if dark_count < 12 or dark_count / float(max(1, near_count)) < 0.08:
        return {"confidence": 0.0, "color": "", "px": 0}

    dark_rgb = tuple(int(round(float(v))) for v in np.median(rgb_crop[dark_ring], axis=0))
    if _luma(dark_rgb) > 105.0:
        return {"confidence": 0.0, "color": "", "px": 0}

    far = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=4).astype(bool) & ~text_mask
    far_dark = far & (gray <= dark_threshold)
    stroke_px = 2 if int(np.count_nonzero(far_dark)) > dark_count * 1.35 else 1
    confidence = min(1.0, max(0.55, 0.5 + dark_count / float(max(1, near_count))))
    return {"confidence": confidence, "color": _hex_color(dark_rgb), "px": stroke_px}


def _detect_light_glow_on_colored_background(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    evidence: TextStyleEvidence,
) -> dict:
    if not evidence.text_color or not _is_light_hex(evidence.text_color):
        return {"confidence": 0.0, "color": "", "px": 0}
    if evidence.stroke_color and _is_dark_hex(evidence.stroke_color):
        return {"confidence": 0.0, "color": "", "px": 0}

    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    if _saturation(background) < 0.22:
        return {"confidence": 0.0, "color": "", "px": 0}

    text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
    text_distance = np.linalg.norm(rgb_crop.astype(np.float32) - text_rgb, axis=2)
    text_mask = text_distance <= 42.0
    text_count = int(np.count_nonzero(text_mask))
    if text_count < 16 or text_count > gray.size * 0.45:
        return {"confidence": 0.0, "color": "", "px": 0}

    kernel = np.ones((3, 3), dtype=np.uint8)
    inner = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    outer = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=6).astype(bool)
    ring = outer & ~inner
    if int(np.count_nonzero(ring)) < 24:
        return {"confidence": 0.0, "color": "", "px": 0}

    bg_luma = _luma(background)
    distance_to_bg = np.linalg.norm(rgb_crop.astype(np.float32) - background, axis=2)
    halo = ring & (distance_to_bg >= 28.0) & (gray >= bg_luma + 8.0)
    halo_count = int(np.count_nonzero(halo))
    ring_count = int(np.count_nonzero(ring))
    if halo_count < 18:
        return {"confidence": 0.90, "color": evidence.text_color, "px": 2}

    halo_ratio = halo_count / float(max(1, ring_count))
    if halo_ratio < 0.10:
        return {"confidence": 0.90, "color": evidence.text_color, "px": 2}

    halo_luma_delta = max(0.0, float(np.median(gray[halo])) - bg_luma)
    confidence = min(1.0, max(0.0, 0.58 + halo_ratio * 0.45 + min(1.0, halo_luma_delta / 70.0) * 0.2))
    return {
        "confidence": confidence,
        "color": evidence.text_color,
        "px": max(2, min(6, int(round(halo_ratio * 8.0)))),
    }


def _detect_shadow_confidence(gray: np.ndarray, contrast: float) -> float:
    rgb = np.dstack([gray, gray, gray]).astype(np.uint8)
    return _detect_shadow_details(rgb, gray, contrast)["confidence"]


def _detect_shadow_details(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    contrast: float,
) -> dict:
    if contrast < 60.0:
        return {"confidence": 0.0, "color": "", "offset": None}

    background_luma = float(np.percentile(gray, 90))
    dark_threshold = float(np.percentile(gray, 5) + max(8.0, contrast * 0.1))
    text_mask = gray <= dark_threshold
    text_count = int(np.count_nonzero(text_mask))
    if text_count < 8 or text_count > gray.size * 0.35:
        return {"confidence": 0.0, "color": "", "offset": None}

    text_luma = float(np.median(gray[text_mask]))
    if background_luma - text_luma < 80.0:
        return {"confidence": 0.0, "color": "", "offset": None}

    shadow_mask = (gray > text_luma + 25.0) & (gray < background_luma - 25.0)
    if int(np.count_nonzero(shadow_mask)) < 8:
        return {"confidence": 0.0, "color": "", "offset": None}

    kernel = np.ones((3, 3), dtype=np.uint8)
    text_guard = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    visible_shadow = shadow_mask & ~text_guard
    if int(np.count_nonzero(visible_shadow)) < 8:
        return {"confidence": 0.0, "color": "", "offset": None}

    best_coverage = 0.0
    best_offset: list[int] | None = None
    for dy in range(-12, 13):
        for dx in range(-12, 13):
            if dx == 0 and dy == 0:
                continue
            shifted = _shift_mask(text_mask, dx, dy) & ~text_guard
            shifted_count = int(np.count_nonzero(shifted))
            if shifted_count < 8:
                continue
            overlap = int(np.count_nonzero(shifted & visible_shadow))
            coverage = overlap / float(shifted_count)
            if coverage > best_coverage:
                best_coverage = coverage
                best_offset = [dx, dy]

    if best_coverage < 0.45:
        return {"confidence": 0.0, "color": "", "offset": None}

    contrast_ratio = min(1.0, (background_luma - float(np.median(gray[visible_shadow]))) / 120.0)
    confidence = min(1.0, max(0.0, 0.35 + best_coverage * 0.45 + contrast_ratio * 0.25))
    if best_offset is None or abs(best_offset[0]) >= 12 or abs(best_offset[1]) >= 12:
        return {"confidence": 0.0, "color": "", "offset": None}
    if abs(best_offset[0]) < 2 or abs(best_offset[1]) < 2:
        return {"confidence": 0.0, "color": "", "offset": None}

    shadow_rgb = tuple(int(round(float(v))) for v in np.median(rgb_crop[visible_shadow], axis=0))
    if max(shadow_rgb) - min(shadow_rgb) > 70:
        return {"confidence": 0.0, "color": "", "offset": None}

    return {
        "confidence": confidence,
        "color": _hex_color(shadow_rgb),
        "offset": best_offset,
    }


def _detect_glow_confidence(gray: np.ndarray, contrast: float) -> float:
    rgb = np.dstack([gray, gray, gray]).astype(np.uint8)
    return _detect_glow_details(rgb, gray, contrast)["confidence"]


def _detect_glow_details(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    contrast: float,
) -> dict:
    if contrast < 70.0:
        return {"confidence": 0.0, "color": "", "px": 0}

    background_luma = float(np.percentile(gray, 20))
    if background_luma > 90.0:
        return {"confidence": 0.0, "color": "", "px": 0}

    light_threshold = float(np.percentile(gray, 95) - max(8.0, contrast * 0.1))
    text_mask = gray >= light_threshold
    text_count = int(np.count_nonzero(text_mask))
    if text_count < 8 or text_count > gray.size * 0.35:
        return {"confidence": 0.0, "color": "", "px": 0}

    text_luma = float(np.median(gray[text_mask]))
    if text_luma - background_luma < 80.0:
        return {"confidence": 0.0, "color": "", "px": 0}
    if _has_adjacent_dark_outline(gray, text_mask, background_luma):
        return {"confidence": 0.0, "color": "", "px": 0}

    kernel = np.ones((3, 3), dtype=np.uint8)
    text_guard = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    halo_outer = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=8).astype(bool)
    halo_inner = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    halo_ring = halo_outer & ~halo_inner
    if int(np.count_nonzero(halo_ring)) < 16:
        return {"confidence": 0.0, "color": "", "px": 0}

    halo_threshold = background_luma + max(18.0, (text_luma - background_luma) * 0.18)
    halo_mask = (gray >= halo_threshold) & halo_ring & ~text_guard
    halo_count = int(np.count_nonzero(halo_mask))
    ring_count = int(np.count_nonzero(halo_ring))
    if halo_count < 16 or ring_count <= 0:
        return {"confidence": 0.0, "color": "", "px": 0}

    halo_ratio = halo_count / float(ring_count)
    halo_luma = float(np.median(gray[halo_mask]))
    halo_delta = halo_luma - background_luma
    if halo_ratio < 0.12 or halo_delta < 18.0:
        return {"confidence": 0.0, "color": "", "px": 0}

    confidence = min(1.0, max(0.0, 0.35 + halo_ratio * 0.5 + min(1.0, halo_delta / 80.0) * 0.25))
    glow_rgb = tuple(int(round(float(v))) for v in np.median(rgb_crop[text_mask], axis=0))
    return {
        "confidence": confidence,
        "color": _hex_color(glow_rgb),
        "px": max(2, min(8, int(round(halo_ratio * 8.0)))),
    }


def _has_adjacent_dark_outline(gray: np.ndarray, text_mask: np.ndarray, background_luma: float) -> bool:
    kernel = np.ones((3, 3), dtype=np.uint8)
    near_ring = cv2.dilate(text_mask.astype(np.uint8), kernel, iterations=2).astype(bool) & ~text_mask
    if int(np.count_nonzero(near_ring)) < 12:
        return False
    dark_threshold = min(background_luma - 18.0, float(np.percentile(gray, 18)))
    dark_ring = near_ring & (gray <= dark_threshold)
    return int(np.count_nonzero(dark_ring)) / float(max(1, int(np.count_nonzero(near_ring)))) >= 0.18


def _detect_gradient_details(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    evidence: TextStyleEvidence,
    contrast: float,
) -> dict:
    if contrast < 45.0:
        return {"confidence": 0.0, "colors": None}

    mask = _foreground_mask_for_evidence(rgb_crop, gray, evidence)
    if int(np.count_nonzero(mask)) < 24:
        return {"confidence": 0.0, "colors": None}

    ys, xs = np.where(mask)
    height_span = int(ys.max() - ys.min() + 1)
    width_span = int(xs.max() - xs.min() + 1)
    if height_span < 8 and width_span < 8:
        return {"confidence": 0.0, "colors": None}

    background = _estimate_border_color(rgb_crop).astype(np.float32)
    best: dict | None = None
    for axis, coords in ((0, ys), (1, xs)):
        lo = float(np.percentile(coords, 25))
        hi = float(np.percentile(coords, 75))
        coord_grid = np.indices(gray.shape)[axis]
        start_mask = mask & (coord_grid <= lo)
        end_mask = mask & (coord_grid >= hi)
        if int(np.count_nonzero(start_mask)) < 8 or int(np.count_nonzero(end_mask)) < 8:
            continue

        start_rgb = np.median(rgb_crop[start_mask], axis=0)
        end_rgb = np.median(rgb_crop[end_mask], axis=0)
        delta = float(np.linalg.norm(start_rgb.astype(np.float32) - end_rgb.astype(np.float32)))
        dark_or_saturated_text = (
            _is_dark_hex(evidence.text_color)
            or _saturation(start_rgb) >= 0.25
            or _saturation(end_rgb) >= 0.25
        )
        min_delta = 32.0 if dark_or_saturated_text else 45.0
        if delta < min_delta:
            continue

        # Avoid treating anti-aliased black/white glyph edges as a gradient.
        start_sat = _saturation(start_rgb)
        end_sat = _saturation(end_rgb)
        if max(start_sat, end_sat) < 0.12 and delta < 95.0:
            continue

        if (
            float(np.linalg.norm(start_rgb.astype(np.float32) - background)) < 55.0
            or float(np.linalg.norm(end_rgb.astype(np.float32) - background)) < 55.0
        ):
            continue

        confidence = min(1.0, max(0.0, 0.45 + delta / 220.0))
        candidate = {
            "confidence": confidence,
            "colors": [_hex_color_tuple(start_rgb), _hex_color_tuple(end_rgb)],
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate

    return best or {"confidence": 0.0, "colors": None}


def _foreground_mask_for_evidence(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    evidence: TextStyleEvidence,
) -> np.ndarray:
    if evidence.stroke_color and evidence.text_color:
        text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
        stroke_rgb = np.array(_parse_hex_color(evidence.stroke_color), dtype=np.float32)
        dist_text = np.linalg.norm(rgb_crop.astype(np.float32) - text_rgb, axis=2)
        dist_stroke = np.linalg.norm(rgb_crop.astype(np.float32) - stroke_rgb, axis=2)
        return (dist_text <= 42.0) & (dist_text + 8.0 <= dist_stroke)

    if evidence.text_color:
        if _is_light_hex(evidence.text_color) or _is_dark_hex(evidence.text_color):
            text_rgb = np.array(_parse_hex_color(evidence.text_color), dtype=np.float32)
            text_distance = np.linalg.norm(rgb_crop.astype(np.float32) - text_rgb, axis=2)
            color_mask = text_distance <= 50.0
            if int(np.count_nonzero(color_mask)) >= 8:
                background = _estimate_border_color(rgb_crop)
                bg_distance = np.linalg.norm(
                    rgb_crop.astype(np.float32) - background.astype(np.float32),
                    axis=2,
                )
                bg_mask = bg_distance > 35.0
                if int(np.count_nonzero(bg_mask)) >= int(np.count_nonzero(color_mask)) * 2:
                    return bg_mask
                return color_mask
        if _is_light_hex(evidence.text_color):
            return gray >= float(np.percentile(gray, 92))
        if _is_dark_hex(evidence.text_color):
            return gray <= float(np.percentile(gray, 12))

    background = _estimate_border_color(rgb_crop)
    distance = np.linalg.norm(rgb_crop.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = distance > 35.0
    if int(np.count_nonzero(mask)) >= 8:
        return mask

    if _is_light_hex(evidence.text_color):
        return gray >= float(np.percentile(gray, 90))
    return gray <= float(np.percentile(gray, 10))


def _detect_curve_details(
    rgb_crop: np.ndarray,
    gray: np.ndarray,
    evidence: TextStyleEvidence,
) -> dict:
    mask = _foreground_mask_for_evidence(rgb_crop, gray, evidence)
    if int(np.count_nonzero(mask)) < 24:
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    ys, xs = np.where(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width < 80 or height < 8:
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    bins = np.linspace(x_min, x_max + 1, 13)
    bin_x: list[float] = []
    bin_y: list[float] = []
    min_points_per_bin = max(3, int(np.count_nonzero(mask) * 0.01))
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = mask & (xs_grid(gray.shape) >= left) & (xs_grid(gray.shape) < right)
        if int(np.count_nonzero(in_bin)) < min_points_per_bin:
            continue
        by, bx = np.where(in_bin)
        bin_x.append(float(np.median(bx)))
        # Use lower ink edge; it tracks the visual baseline better for blocky glyphs.
        bin_y.append(float(np.percentile(by, 85)))

    if len(bin_x) < 6:
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    x_arr = np.array(bin_x, dtype=np.float32)
    y_arr = np.array(bin_y, dtype=np.float32)
    x_norm = (x_arr - float(np.mean(x_arr))) / max(1.0, float(np.ptp(x_arr)) / 2.0)

    line_coeff = np.polyfit(x_norm, y_arr, 1)
    quad_coeff = np.polyfit(x_norm, y_arr, 2)
    line_pred = np.polyval(line_coeff, x_norm)
    quad_pred = np.polyval(quad_coeff, x_norm)
    line_rmse = float(np.sqrt(np.mean((y_arr - line_pred) ** 2)))
    quad_rmse = float(np.sqrt(np.mean((y_arr - quad_pred) ** 2)))
    improvement = line_rmse - quad_rmse

    center_y = float(np.polyval(quad_coeff, 0.0))
    edge_y = float((np.polyval(quad_coeff, -1.0) + np.polyval(quad_coeff, 1.0)) / 2.0)
    curve_px = edge_y - center_y
    amount = abs(curve_px) / max(1.0, float(height))
    if amount < 0.10 or improvement < 3.0:
        asymmetric = _detect_asymmetric_curve_from_bins(x_norm, y_arr, height)
        if asymmetric["confidence"] >= 0.60:
            return asymmetric
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    direction = "arc_up" if curve_px > 0 else "arc_down"
    confidence = min(1.0, max(0.0, 0.45 + amount * 0.9 + min(1.0, improvement / 18.0) * 0.25))
    return {
        "confidence": confidence,
        "direction": direction,
        "amount": min(1.0, amount),
    }


def _detect_asymmetric_curve_from_bins(
    x_norm: np.ndarray,
    y_arr: np.ndarray,
    height: int,
) -> dict:
    if len(x_norm) < 8:
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    line_coeff = np.polyfit(x_norm, y_arr, 1)
    cubic_coeff = np.polyfit(x_norm, y_arr, 3)
    line_pred = np.polyval(line_coeff, x_norm)
    cubic_pred = np.polyval(cubic_coeff, x_norm)
    line_rmse = float(np.sqrt(np.mean((y_arr - line_pred) ** 2)))
    cubic_rmse = float(np.sqrt(np.mean((y_arr - cubic_pred) ** 2)))
    improvement = line_rmse - cubic_rmse

    mid = len(x_norm) // 2
    left_slope = float(np.polyfit(x_norm[: mid + 1], y_arr[: mid + 1], 1)[0])
    right_slope = float(np.polyfit(x_norm[mid:], y_arr[mid:], 1)[0])
    slope_delta = right_slope - left_slope
    amount = max(
        abs(slope_delta) / max(1.0, float(height) * 2.0),
        line_rmse / max(1.0, float(height)),
    )

    if line_rmse < 6.0 or improvement < 4.0 or abs(slope_delta) < 7.0 or amount < 0.06:
        return {"confidence": 0.0, "direction": "", "amount": 0.0}

    direction = "arc_up" if slope_delta > 0 else "arc_down"
    confidence = min(1.0, max(0.0, 0.45 + amount * 0.9 + min(1.0, improvement / 14.0) * 0.25))
    return {
        "confidence": confidence,
        "direction": direction,
        "amount": min(1.0, amount),
    }


def xs_grid(shape: tuple[int, int]) -> np.ndarray:
    return np.indices(shape)[1]


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape[:2]

    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(width, width - dx)
    src_y2 = min(height, height - dy)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return shifted

    dst_x1 = src_x1 + dx
    dst_y1 = src_y1 + dy
    dst_x2 = src_x2 + dx
    dst_y2 = src_y2 + dy
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _hex_color_tuple(rgb: np.ndarray) -> str:
    values = tuple(int(round(float(v))) for v in rgb[:3])
    values = tuple(max(0, min(255, v)) for v in values)
    return _hex_color(values)  # type: ignore[arg-type]


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    value = str(value or "").strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    except ValueError:
        return (0, 0, 0)


def _colors_close_hex(left: str, right: str, *, tolerance: float) -> bool:
    left_rgb = np.array(_parse_hex_color(left), dtype=np.float32)
    right_rgb = np.array(_parse_hex_color(right), dtype=np.float32)
    return float(np.linalg.norm(left_rgb - right_rgb)) <= tolerance


def _color_close_to_background(
    rgb: tuple[int, int, int],
    rgb_crop: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    background = _estimate_border_color(rgb_crop[:, :, :3]).astype(np.float32)
    color = np.array(rgb, dtype=np.float32)
    return float(np.linalg.norm(color - background)) <= tolerance


def _luma(rgb: np.ndarray | tuple[int, int, int]) -> float:
    arr = np.array(rgb, dtype=np.float32)
    return float(arr[0] * 0.299 + arr[1] * 0.587 + arr[2] * 0.114)


def _is_light_hex(value: str) -> bool:
    return _luma(_parse_hex_color(value)) >= 220.0


def _is_dark_hex(value: str) -> bool:
    return _luma(_parse_hex_color(value)) <= 45.0


def _saturation(rgb: np.ndarray) -> float:
    values = np.array(rgb[:3], dtype=np.float32) / 255.0
    max_v = float(np.max(values))
    min_v = float(np.min(values))
    if max_v <= 0.0:
        return 0.0
    return (max_v - min_v) / max_v


def _estimate_border_color(rgb_crop: np.ndarray) -> np.ndarray:
    return np.median(
        np.concatenate(
            [
                rgb_crop[:2].reshape(-1, 3),
                rgb_crop[-2:].reshape(-1, 3),
                rgb_crop[:, :2].reshape(-1, 3),
                rgb_crop[:, -2:].reshape(-1, 3),
            ],
            axis=0,
        ),
        axis=0,
    )


def _empty() -> TextStyleEvidence:
    return TextStyleEvidence(
        source="none",
        text_color="",
        text_color_confidence=0.0,
        stroke_color="",
        stroke_width_px=0,
        stroke_confidence=0.0,
        shadow=False,
        shadow_confidence=0.0,
        glow=False,
        glow_confidence=0.0,
        font_name="",
        font_confidence=0.0,
    )
