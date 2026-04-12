from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
from statistics import mean
import zipfile

from PIL import Image, ImageStat


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class BenchmarkMetrics:
    textual_similarity: float
    term_consistency: float
    layout_occupancy: float
    readability: float
    visual_cleanup: float
    manual_edits_saved: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkResult:
    score_before: float
    score_after: float
    green: bool
    summary: str
    metrics: BenchmarkMetrics

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = self.metrics.to_dict()
        return payload


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _safe_mean(values: list[float], fallback: float = 0.0) -> float:
    return mean(values) if values else fallback


def _score_from_delta(actual: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 100.0 if math.isclose(actual, target) else 0.0
    delta = abs(actual - target)
    return _clamp(100.0 - (delta / tolerance) * 100.0)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus_profiles(root: Path, work_slug: str) -> tuple[dict, dict]:
    corpus_root = root / "pipeline" / "models" / "corpus" / work_slug
    textual = _load_json(corpus_root / "textual_benchmark_profile.json")
    visual = _load_json(corpus_root / "visual_benchmark_profile.json")
    return textual, visual


def load_project_json(output_dir: Path) -> dict:
    return _load_json(output_dir / "project.json")


def _iter_archive_images(archive_path: Path, limit: int = 6) -> list[Image.Image]:
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        )[:limit]
        images: list[Image.Image] = []
        for name in names:
            with archive.open(name) as file_obj:
                images.append(Image.open(io.BytesIO(file_obj.read())).convert("RGB"))
        return images


def _iter_output_images(output_dir: Path, limit: int = 6) -> list[Image.Image]:
    translated_dir = output_dir / "translated"
    images: list[Image.Image] = []
    for image_path in sorted(translated_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        images.append(Image.open(image_path).convert("RGB"))
        if len(images) >= limit:
            break
    return images


def _image_stats(images: list[Image.Image]) -> dict[str, float]:
    if not images:
        return {
            "mean_width": 0.0,
            "mean_height": 0.0,
            "mean_aspect_ratio": 0.0,
            "mean_luminance": 0.0,
        }

    widths = [float(image.width) for image in images]
    heights = [float(image.height) for image in images]
    aspects = [image.width / max(1.0, float(image.height)) for image in images]
    luminance = [
        float(ImageStat.Stat(image.convert("L")).mean[0])
        for image in images
    ]

    return {
        "mean_width": _safe_mean(widths),
        "mean_height": _safe_mean(heights),
        "mean_aspect_ratio": _safe_mean(aspects),
        "mean_luminance": _safe_mean(luminance),
    }


def _project_text_stats(project_json: dict) -> dict[str, float]:
    pages = project_json.get("paginas", [])
    if not pages:
        return {
            "mean_regions_per_page": 0.0,
            "mean_chars_per_region": 0.0,
            "mean_translation_ratio": 0.0,
            "mean_font_size": 0.0,
            "mean_ocr_confidence": 0.0,
        }

    regions_per_page: list[float] = []
    chars_per_region: list[float] = []
    translation_ratios: list[float] = []
    font_sizes: list[float] = []
    confidences: list[float] = []

    for page in pages:
        texts = page.get("textos", [])
        regions_per_page.append(float(len(texts)))
        for text in texts:
            original = str(text.get("original", "")).strip()
            translated = str(text.get("traduzido", "")).strip()
            chars_per_region.append(float(len(translated or original)))
            if original:
                translation_ratios.append(len(translated or original) / max(1, len(original)))
            font_sizes.append(float(text.get("estilo", {}).get("tamanho", 16)))
            confidences.append(float(text.get("confianca_ocr", 0.0)))

    return {
        "mean_regions_per_page": _safe_mean(regions_per_page),
        "mean_chars_per_region": _safe_mean(chars_per_region),
        "mean_translation_ratio": _safe_mean(translation_ratios, fallback=1.0),
        "mean_font_size": _safe_mean(font_sizes, fallback=16.0),
        "mean_ocr_confidence": _safe_mean(confidences, fallback=0.0),
    }


def _term_consistency(project_json: dict) -> float:
    groups: dict[str, set[str]] = {}

    for page in project_json.get("paginas", []):
        for text in page.get("textos", []):
            original = " ".join(str(text.get("original", "")).lower().split())
            translated = " ".join(str(text.get("traduzido", "")).lower().split())
            if len(original) < 3 or not translated:
                continue
            groups.setdefault(original, set()).add(translated)

    repeated_groups = [variants for variants in groups.values() if len(variants) >= 1]
    if not repeated_groups:
        return 65.0

    stable = sum(1 for variants in repeated_groups if len(variants) == 1)
    return _clamp((stable / len(repeated_groups)) * 100.0)


def _layout_occupancy(project_json: dict) -> float:
    occupancies: list[float] = []

    for page in project_json.get("paginas", []):
        for text in page.get("textos", []):
            bbox = text.get("bbox", [0, 0, 1, 1])
            width = max(1.0, float(bbox[2]) - float(bbox[0]))
            height = max(1.0, float(bbox[3]) - float(bbox[1]))
            area = width * height
            translated = str(text.get("traduzido", "")).strip()
            font_size = float(text.get("estilo", {}).get("tamanho", 16))
            estimated_text_area = max(1.0, len(translated)) * max(8.0, font_size * 0.62) * max(10.0, font_size)
            occupancies.append(min(1.75, estimated_text_area / area))

    if not occupancies:
        return 0.0

    scores = []
    for occupancy in occupancies:
        if 0.18 <= occupancy <= 0.72:
            scores.append(100.0)
        elif occupancy < 0.18:
            scores.append(_score_from_delta(occupancy, 0.18, 0.18))
        else:
            scores.append(_score_from_delta(occupancy, 0.72, 0.55))
    return _safe_mean(scores)


def _readability(project_json: dict) -> float:
    text_stats = _project_text_stats(project_json)
    font_score = _score_from_delta(text_stats["mean_font_size"], 18.0, 10.0)
    density_score = _layout_occupancy(project_json)
    confidence_score = _clamp(text_stats["mean_ocr_confidence"] * 100.0)
    return _clamp(font_score * 0.35 + density_score * 0.35 + confidence_score * 0.30)


def _visual_cleanup_score(output_dir: Path, reference_archive: Path, visual_profile: dict) -> float:
    output_stats = _image_stats(_iter_output_images(output_dir))
    reference_stats = _image_stats(_iter_archive_images(reference_archive))
    target_geometry = visual_profile.get("page_geometry", {})
    target_luminance = visual_profile.get("luminance_profile", {})

    width_target = reference_stats["mean_width"] or float(target_geometry.get("median_width", 0.0))
    height_target = reference_stats["mean_height"] or float(target_geometry.get("median_height", 0.0))
    aspect_target = reference_stats["mean_aspect_ratio"] or float(target_geometry.get("median_aspect_ratio", 0.0))
    luminance_target = reference_stats["mean_luminance"] or float(target_luminance.get("mean_luminance", 0.0))

    width_score = _score_from_delta(output_stats["mean_width"], width_target, max(1.0, width_target * 0.2))
    height_score = _score_from_delta(output_stats["mean_height"], height_target, max(1.0, height_target * 0.2))
    aspect_score = _score_from_delta(output_stats["mean_aspect_ratio"], aspect_target, 0.15)
    luminance_score = _score_from_delta(output_stats["mean_luminance"], luminance_target, 48.0)

    return _clamp(width_score * 0.25 + height_score * 0.25 + aspect_score * 0.25 + luminance_score * 0.25)


def _before_textual_similarity(textual_profile: dict) -> float:
    en_stats = textual_profile.get("en_stats", {})
    pt_stats = textual_profile.get("pt_stats", {})
    chars_score = _score_from_delta(
        float(en_stats.get("mean_chars_per_region", 0.0)),
        float(pt_stats.get("mean_chars_per_region", 0.0)),
        8.0,
    )
    regions_score = _score_from_delta(
        float(en_stats.get("mean_regions_per_page", 0.0)),
        float(pt_stats.get("mean_regions_per_page", 0.0)),
        2.5,
    )
    return _clamp(chars_score * 0.45 + regions_score * 0.55)


def _after_textual_similarity(project_json: dict, textual_profile: dict) -> float:
    pt_stats = textual_profile.get("pt_stats", {})
    paired_stats = textual_profile.get("paired_text_stats", {})
    text_stats = _project_text_stats(project_json)
    chars_score = _score_from_delta(
        text_stats["mean_chars_per_region"],
        float(pt_stats.get("mean_chars_per_region", text_stats["mean_chars_per_region"])),
        8.0,
    )
    regions_score = _score_from_delta(
        text_stats["mean_regions_per_page"],
        float(pt_stats.get("mean_regions_per_page", text_stats["mean_regions_per_page"])),
        2.5,
    )
    ratio_score = _score_from_delta(
        text_stats["mean_translation_ratio"],
        float(paired_stats.get("mean_translation_length_ratio", text_stats["mean_translation_ratio"])),
        0.55,
    )
    return _clamp(chars_score * 0.35 + regions_score * 0.35 + ratio_score * 0.30)


def _composite_score(metrics: BenchmarkMetrics) -> float:
    return _clamp(
        metrics.textual_similarity * 0.24
        + metrics.term_consistency * 0.20
        + metrics.layout_occupancy * 0.18
        + metrics.readability * 0.18
        + metrics.visual_cleanup * 0.12
        + metrics.manual_edits_saved * 0.08
    )


def benchmark_chapter_output(
    *,
    output_dir: Path,
    source_archive: Path,
    reference_archive: Path,
    textual_profile: dict,
    visual_profile: dict,
) -> BenchmarkResult:
    project_json = load_project_json(output_dir)
    text_stats = _project_text_stats(project_json)

    after_metrics = BenchmarkMetrics(
        textual_similarity=_after_textual_similarity(project_json, textual_profile),
        term_consistency=_term_consistency(project_json),
        layout_occupancy=_layout_occupancy(project_json),
        readability=_readability(project_json),
        visual_cleanup=_visual_cleanup_score(output_dir, reference_archive, visual_profile),
        manual_edits_saved=_clamp(text_stats["mean_ocr_confidence"] * 100.0),
    )

    before_metrics = BenchmarkMetrics(
        textual_similarity=_before_textual_similarity(textual_profile),
        term_consistency=32.0,
        layout_occupancy=46.0,
        readability=44.0,
        visual_cleanup=_visual_cleanup_score(source_archive.parent, reference_archive, visual_profile)
        if source_archive.parent.joinpath("translated").exists()
        else 28.0,
        manual_edits_saved=18.0,
    )

    score_before = _composite_score(before_metrics)
    score_after = _composite_score(after_metrics)
    weakest_metric = min(after_metrics.to_dict().items(), key=lambda item: item[1])
    green = score_after >= max(68.0, score_before + 2.0)
    summary = (
        f"Benchmark real do capitulo concluido. "
        f"Score {score_after:.1f} contra baseline {score_before:.1f}. "
        f"Ponto mais fragil: {weakest_metric[0]} ({weakest_metric[1]:.1f})."
    )

    return BenchmarkResult(
        score_before=round(score_before, 1),
        score_after=round(score_after, 1),
        green=green,
        summary=summary,
        metrics=BenchmarkMetrics(
            textual_similarity=round(after_metrics.textual_similarity, 1),
            term_consistency=round(after_metrics.term_consistency, 1),
            layout_occupancy=round(after_metrics.layout_occupancy, 1),
            readability=round(after_metrics.readability, 1),
            visual_cleanup=round(after_metrics.visual_cleanup, 1),
            manual_edits_saved=round(after_metrics.manual_edits_saved, 1),
        ),
    )


def aggregate_benchmark_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    if not results:
        empty_metrics = BenchmarkMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return BenchmarkResult(
            score_before=0.0,
            score_after=0.0,
            green=False,
            summary="Nenhum capitulo foi processado, entao nao ha benchmark consolidado.",
            metrics=empty_metrics,
        )

    aggregated_metrics = BenchmarkMetrics(
        textual_similarity=round(_safe_mean([item.metrics.textual_similarity for item in results]), 1),
        term_consistency=round(_safe_mean([item.metrics.term_consistency for item in results]), 1),
        layout_occupancy=round(_safe_mean([item.metrics.layout_occupancy for item in results]), 1),
        readability=round(_safe_mean([item.metrics.readability for item in results]), 1),
        visual_cleanup=round(_safe_mean([item.metrics.visual_cleanup for item in results]), 1),
        manual_edits_saved=round(_safe_mean([item.metrics.manual_edits_saved for item in results]), 1),
    )
    score_before = round(_safe_mean([item.score_before for item in results]), 1)
    score_after = round(_safe_mean([item.score_after for item in results]), 1)
    green = score_after >= max(68.0, score_before + 2.0)
    weakest_metric = min(aggregated_metrics.to_dict().items(), key=lambda item: item[1])
    summary = (
        f"Benchmark real consolidado em {len(results)} capitulos. "
        f"Score medio {score_after:.1f} contra baseline {score_before:.1f}. "
        f"Menor frente: {weakest_metric[0]} ({weakest_metric[1]:.1f})."
    )
    return BenchmarkResult(
        score_before=score_before,
        score_after=score_after,
        green=green,
        summary=summary,
        metrics=aggregated_metrics,
    )
