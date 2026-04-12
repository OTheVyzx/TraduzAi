from __future__ import annotations

import json
import re
import statistics
import tempfile
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageStat


PT_PATTERN = re.compile(r"^(?P<source>.+?)_Cap[ií]tulo\s+(?P<chapter>\d+)", re.IGNORECASE)
EN_PATTERN = re.compile(r"^Chapter\s+(?P<chapter>\d+)", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
HASH_BITS = 64
DEFAULT_GAP_COST = 0.35

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def parse_pt_chapter_filename(name: str) -> dict:
    match = PT_PATTERN.search(Path(name).name)
    if not match:
        raise ValueError(f"Nome PT-BR invalido: {name}")
    return {
        "chapter": int(match.group("chapter")),
        "source_group": match.group("source").strip(),
    }


def parse_en_chapter_filename(name: str) -> dict:
    match = EN_PATTERN.search(Path(name).name)
    if not match:
        raise ValueError(f"Nome EN invalido: {name}")
    return {"chapter": int(match.group("chapter"))}


def scan_pt_chapters(directory: str | Path) -> list[dict]:
    entries = []
    for path in sorted(Path(directory).glob("*.cbz")):
        parsed = parse_pt_chapter_filename(path.name)
        entries.append(
            {
                "chapter": parsed["chapter"],
                "path": str(path),
                "source_group": parsed["source_group"],
            }
        )
    return entries


def scan_en_chapters(directory: str | Path) -> list[dict]:
    entries = []
    for path in sorted(Path(directory).glob("*.cbz")):
        parsed = parse_en_chapter_filename(path.name)
        entries.append({"chapter": parsed["chapter"], "path": str(path)})
    return entries


def pair_parallel_chapters(pt_entries: list[dict], en_entries: list[dict]) -> list[dict]:
    en_by_chapter = {entry["chapter"]: entry for entry in en_entries}
    pairs = []
    for pt_entry in pt_entries:
        en_entry = en_by_chapter.get(pt_entry["chapter"])
        if not en_entry:
            continue
        pairs.append(
            {
                "chapter": pt_entry["chapter"],
                "pt_path": pt_entry["path"],
                "en_path": en_entry["path"],
                "source_group": pt_entry["source_group"],
            }
        )
    return sorted(pairs, key=lambda item: item["chapter"])


def _list_cbz_image_infos(path: str | Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path, "r") as archive:
        infos = [
            info
            for info in archive.infolist()
            if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES and not info.is_dir()
        ]
    return sorted(infos, key=lambda info: info.filename)


def count_cbz_pages(path: str | Path) -> int:
    return len(_list_cbz_image_infos(path))


def iter_cbz_images(path: str | Path, limit: int | None = None):
    infos = _list_cbz_image_infos(path)
    with zipfile.ZipFile(path, "r") as archive:
        for index, info in enumerate(infos):
            if limit is not None and index >= limit:
                break
            with archive.open(info, "r") as image_file:
                yield info.filename, Image.open(BytesIO(image_file.read())).convert("RGB")


def _extract_cbz_page(cbz_path: str | Path, page_number: int, destination_path: str | Path):
    infos = _list_cbz_image_infos(cbz_path)
    if page_number < 1 or page_number > len(infos):
        raise IndexError(f"Pagina fora do intervalo: {page_number}")
    info = infos[page_number - 1]
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cbz_path, "r") as archive:
        with archive.open(info, "r") as image_file:
            destination.write_bytes(image_file.read())
    return str(destination)


def build_manifest(work_slug: str, pt_entries: list[dict], en_entries: list[dict], pairs: list[dict]) -> dict:
    paired = []
    for pair in pairs:
        paired.append(
            {
                "chapter": pair["chapter"],
                "pt_path": pair["pt_path"],
                "en_path": pair["en_path"],
                "source_group": pair["source_group"],
                "pt_pages": count_cbz_pages(pair["pt_path"]),
                "en_pages": count_cbz_pages(pair["en_path"]),
            }
        )

    return {
        "work_slug": work_slug,
        "total_pt_chapters": len(pt_entries),
        "total_en_chapters": len(en_entries),
        "total_paired_chapters": len(paired),
        "paired_chapters": paired,
    }


def build_quality_profile(manifest: dict) -> dict:
    distribution: dict[str, int] = {}
    total_pt_pages = 0
    for item in manifest.get("paired_chapters", []):
        distribution[item["source_group"]] = distribution.get(item["source_group"], 0) + 1
        total_pt_pages += int(item["pt_pages"])

    chapters = [item["chapter"] for item in manifest.get("paired_chapters", [])]
    return {
        "work_slug": manifest["work_slug"],
        "total_paired_chapters": manifest["total_paired_chapters"],
        "total_pt_pages": total_pt_pages,
        "chapter_range": [min(chapters), max(chapters)] if chapters else [],
        "pt_source_distribution": distribution,
    }


def build_alignment_profile(manifest: dict) -> dict:
    page_deltas = []
    for item in manifest.get("paired_chapters", []):
        page_deltas.append(
            {
                "chapter": item["chapter"],
                "pt_pages": item["pt_pages"],
                "en_pages": item["en_pages"],
                "page_delta": item["pt_pages"] - item["en_pages"],
            }
        )
    return {
        "work_slug": manifest["work_slug"],
        "total_paired_chapters": manifest["total_paired_chapters"],
        "page_deltas": page_deltas,
        "ready_for_segment_alignment": bool(page_deltas),
    }


def _build_source_distribution(manifest: dict) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in manifest.get("paired_chapters", []):
        distribution[item["source_group"]] = distribution.get(item["source_group"], 0) + 1
    return distribution


def build_work_profile(manifest: dict) -> dict:
    chapters = [item["chapter"] for item in manifest.get("paired_chapters", [])]
    total_pt_pages = sum(int(item["pt_pages"]) for item in manifest.get("paired_chapters", []))
    total_en_pages = sum(int(item["en_pages"]) for item in manifest.get("paired_chapters", []))

    return {
        "work_slug": manifest["work_slug"],
        "chapter_range": [min(chapters), max(chapters)] if chapters else [],
        "paired_totals": {
            "chapters": manifest["total_paired_chapters"],
            "pt_pages": total_pt_pages,
            "en_pages": total_en_pages,
            "page_delta": total_pt_pages - total_en_pages,
        },
        "provenance": {
            "pt_source_distribution": _build_source_distribution(manifest),
        },
    }


def _classify_luminance(mean_luminance: float) -> str:
    if mean_luminance >= 220:
        return "light"
    if mean_luminance <= 75:
        return "dark"
    return "mid"


def build_visual_benchmark_profile(manifest: dict, max_pages_per_chapter: int = 6) -> dict:
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    luminance_values: list[float] = []
    luminance_buckets = {"light": 0, "mid": 0, "dark": 0}

    for item in manifest.get("paired_chapters", []):
        for _, image in iter_cbz_images(item["pt_path"], limit=max_pages_per_chapter):
            width, height = image.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(round(width / height, 4) if height else 0.0)
            mean_luminance = float(ImageStat.Stat(image.convert("L")).mean[0])
            luminance_values.append(mean_luminance)
            luminance_buckets[_classify_luminance(mean_luminance)] += 1

    if not widths:
        return {
            "work_slug": manifest["work_slug"],
            "sampled_pages": 0,
            "page_geometry": {},
            "luminance_profile": {
                "light_pages": 0,
                "mid_pages": 0,
                "dark_pages": 0,
                "mean_luminance": 0.0,
            },
        }

    return {
        "work_slug": manifest["work_slug"],
        "sampled_pages": len(widths),
        "page_geometry": {
            "median_width": int(statistics.median(widths)),
            "median_height": int(statistics.median(heights)),
            "median_aspect_ratio": round(statistics.median(aspect_ratios), 4),
        },
        "luminance_profile": {
            "light_pages": luminance_buckets["light"],
            "mid_pages": luminance_buckets["mid"],
            "dark_pages": luminance_buckets["dark"],
            "mean_luminance": round(statistics.mean(luminance_values), 2),
        },
    }


def _compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), RESAMPLE_LANCZOS)
    pixels = list(grayscale.tobytes())
    bit_string = 0
    for row in range(hash_size):
        row_offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            bit_string = (bit_string << 1) | int(left > right)
    return bit_string


def _hamming_distance(lhs: int, rhs: int) -> int:
    return (lhs ^ rhs).bit_count()


def _page_hashes(cbz_path: str | Path) -> list[dict]:
    hashes = []
    for page_index, (_, image) in enumerate(iter_cbz_images(cbz_path), start=1):
        hashes.append({"page": page_index, "hash": _compute_dhash(image)})
    return hashes


def _align_page_hashes(pt_hashes: list[dict], en_hashes: list[dict], gap_cost: float = DEFAULT_GAP_COST) -> list[dict]:
    n = len(pt_hashes)
    m = len(en_hashes)
    if not n or not m:
        return []

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        trace[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        trace[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance = _hamming_distance(pt_hashes[i - 1]["hash"], en_hashes[j - 1]["hash"]) / HASH_BITS
            candidates = [
                (dp[i - 1][j - 1] + distance, "diag"),
                (dp[i - 1][j] + gap_cost, "up"),
                (dp[i][j - 1] + gap_cost, "left"),
            ]
            best_cost, best_op = min(candidates, key=lambda item: item[0])
            dp[i][j] = best_cost
            trace[i][j] = best_op

    matches = []
    i, j = n, m
    while i > 0 or j > 0:
        op = trace[i][j]
        if op == "diag":
            distance = _hamming_distance(pt_hashes[i - 1]["hash"], en_hashes[j - 1]["hash"]) / HASH_BITS
            matches.append(
                {
                    "pt_page": pt_hashes[i - 1]["page"],
                    "en_page": en_hashes[j - 1]["page"],
                    "distance": round(distance, 4),
                }
            )
            i -= 1
            j -= 1
        elif op == "up":
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def build_page_alignment_profile(manifest: dict, gap_cost: float = DEFAULT_GAP_COST) -> dict:
    chapters = []
    for item in manifest.get("paired_chapters", []):
        pt_hashes = _page_hashes(item["pt_path"])
        en_hashes = _page_hashes(item["en_path"])
        mappings = _align_page_hashes(pt_hashes, en_hashes, gap_cost=gap_cost)
        average_distance = round(
            statistics.mean(mapping["distance"] for mapping in mappings),
            4,
        ) if mappings else 1.0
        coverage_ratio = round(
            len(mappings) / max(1, min(len(pt_hashes), len(en_hashes))),
            4,
        )
        chapters.append(
            {
                "chapter": item["chapter"],
                "pt_pages": len(pt_hashes),
                "en_pages": len(en_hashes),
                "matched_pages": len(mappings),
                "coverage_ratio": coverage_ratio,
                "average_distance": average_distance,
                "mappings": mappings,
            }
        )

    return {
        "work_slug": manifest["work_slug"],
        "total_paired_chapters": manifest["total_paired_chapters"],
        "chapters": chapters,
        "ready_for_text_alignment": any(chapter["matched_pages"] for chapter in chapters),
    }


def _downsample_evenly(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)

    selected = []
    for index in range(limit):
        item_index = round(index * (len(items) - 1) / max(1, limit - 1))
        selected.append(items[item_index])
    return selected


def select_aligned_page_samples(manifest: dict, page_alignment_profile: dict, max_page_pairs: int = 24) -> list[dict]:
    if max_page_pairs <= 0:
        return []

    manifest_by_chapter = {item["chapter"]: item for item in manifest.get("paired_chapters", [])}
    primary_samples = []
    secondary_samples = []

    for chapter in page_alignment_profile.get("chapters", []):
        mappings = chapter.get("mappings", [])
        if not mappings:
            continue
        manifest_item = manifest_by_chapter.get(chapter["chapter"])
        if not manifest_item:
            continue

        indices = [len(mappings) // 2]
        if len(mappings) >= 4:
            indices.extend([len(mappings) // 4, (len(mappings) * 3) // 4])

        chosen = []
        seen = set()
        for idx in indices:
            idx = max(0, min(idx, len(mappings) - 1))
            mapping = mappings[idx]
            key = (mapping["pt_page"], mapping["en_page"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(
                {
                    "chapter": chapter["chapter"],
                    "pt_path": manifest_item["pt_path"],
                    "en_path": manifest_item["en_path"],
                    "pt_page": mapping["pt_page"],
                    "en_page": mapping["en_page"],
                    "distance": mapping["distance"],
                }
            )
        if chosen:
            primary_samples.append(chosen[0])
            secondary_samples.extend(chosen[1:])

    selected = _downsample_evenly(primary_samples, min(max_page_pairs, len(primary_samples)))
    remaining = max_page_pairs - len(selected)
    if remaining > 0 and secondary_samples:
        selected.extend(_downsample_evenly(secondary_samples, remaining))
    return selected


def _default_ocr_runner(image_path: str) -> dict:
    from ocr.detector import run_ocr

    return run_ocr(image_path=image_path, profile="quality")


def _sort_text_regions(texts: list[dict]) -> list[dict]:
    def key(item: dict):
        bbox = item.get("bbox") or [0, 0, 0, 0]
        center_y = (bbox[1] + bbox[3]) / 2
        return (round(center_y, 2), bbox[0])

    return sorted(texts, key=key)


def _normalize_memory_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip(" \t\r\n'\"“”‘’.,!?;:()[]{}")


def _is_valid_memory_text(text: str) -> bool:
    normalized = _normalize_memory_text(text)
    return len(normalized) >= 2 and bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", normalized))


def _alpha_ratio(text: str) -> float:
    normalized = _normalize_memory_text(text)
    if not normalized:
        return 0.0
    alpha_chars = sum(1 for char in normalized if char.isalpha())
    return alpha_chars / len(normalized)


def _is_valid_memory_pair(source_item: dict, target_item: dict, position_delta: float) -> bool:
    source_text = _normalize_memory_text(source_item.get("text", ""))
    target_text = _normalize_memory_text(target_item.get("text", ""))
    if not (_is_valid_memory_text(source_text) and _is_valid_memory_text(target_text)):
        return False
    if position_delta > 0.12:
        return False
    if _alpha_ratio(source_text) < 0.55 or _alpha_ratio(target_text) < 0.55:
        return False

    source_confidence = float(source_item.get("confidence", 1.0) or 0.0)
    target_confidence = float(target_item.get("confidence", 1.0) or 0.0)
    if min(source_confidence, target_confidence) < 0.35:
        return False

    combined = f"{source_text} {target_text}".casefold()
    if any(marker in combined for marker in ("scan", "www", ".com", "asura", "mangaflix", "worldscan")):
        return False

    source_tokens = len(WORD_PATTERN.findall(source_text))
    target_tokens = len(WORD_PATTERN.findall(target_text))
    if abs(source_tokens - target_tokens) > 2 and max(source_tokens, target_tokens) >= 3:
        return False
    length_ratio = len(target_text) / max(1, len(source_text))
    if length_ratio < 0.45 or length_ratio > 2.4:
        return False

    source_type = source_item.get("tipo") or ""
    target_type = target_item.get("tipo") or ""
    if "sfx" in {source_type, target_type}:
        return False
    if source_type and target_type and source_type != target_type:
        return False
    return True


def _pair_text_regions(en_texts: list[dict], pt_texts: list[dict]) -> list[dict]:
    source = _sort_text_regions(en_texts)
    target = _sort_text_regions(pt_texts)
    if not source or not target:
        return []

    used_targets = set()
    max_y = max(
        [bbox for item in source + target for bbox in (item.get("bbox") or [0, 0, 0, 0])[1::2]] or [1]
    )
    pairs = []
    for index, source_item in enumerate(source):
        if len(target) == 1:
            preferred_index = 0
        else:
            preferred_index = round(index * (len(target) - 1) / max(1, len(source) - 1))

        candidate_indices = sorted(
            range(len(target)),
            key=lambda target_index: (
                abs(target_index - preferred_index),
                abs(_center_y(source_item) - _center_y(target[target_index])),
            ),
        )
        target_index = next((idx for idx in candidate_indices if idx not in used_targets), None)
        if target_index is None:
            break
        used_targets.add(target_index)
        target_item = target[target_index]
        position_delta = abs(_center_y(source_item) - _center_y(target_item)) / max_y
        pairs.append(
            {
                "source": source_item,
                "target": target_item,
                "position_delta": round(position_delta, 4),
            }
        )
    return pairs


def _center_y(item: dict) -> float:
    bbox = item.get("bbox") or [0, 0, 0, 0]
    return (bbox[1] + bbox[3]) / 2


def collect_ocr_sample_pairs(
    sample_pairs: list[dict],
    ocr_runner=None,
) -> list[dict]:
    runner = ocr_runner or _default_ocr_runner
    collected = []
    for sample in sample_pairs:
        en_result = runner(sample["en_image_path"])
        pt_result = runner(sample["pt_image_path"])
        collected.append(
            {
                **sample,
                "en_result": en_result,
                "pt_result": pt_result,
                "paired_regions": _pair_text_regions(
                    en_result.get("texts", []),
                    pt_result.get("texts", []),
                ),
            }
        )
    return collected


def build_translation_memory_candidates_from_ocr_samples(
    ocr_samples: list[dict],
    work_slug: str,
) -> dict:
    candidate_map: dict[tuple[str, str], dict] = {}

    for sample in ocr_samples:
        chapter = sample["chapter"]
        for pair in sample.get("paired_regions", []):
            if not _is_valid_memory_pair(pair["source"], pair["target"], pair["position_delta"]):
                continue
            source_text = _normalize_memory_text(pair["source"].get("text", ""))
            target_text = _normalize_memory_text(pair["target"].get("text", ""))

            key = (source_text.casefold(), target_text.casefold())
            entry = candidate_map.setdefault(
                key,
                {
                    "source_text": source_text,
                    "target_text": target_text,
                    "occurrences": 0,
                    "chapters": set(),
                    "total_position_delta": 0.0,
                    "source_tokens": len(WORD_PATTERN.findall(source_text)),
                    "target_tokens": len(WORD_PATTERN.findall(target_text)),
                },
            )
            entry["occurrences"] += 1
            entry["chapters"].add(chapter)
            entry["total_position_delta"] += pair["position_delta"]

    candidates = []
    glossary_candidates = []
    for entry in candidate_map.values():
        candidate = {
            "source_text": entry["source_text"],
            "target_text": entry["target_text"],
            "occurrences": entry["occurrences"],
            "chapters": sorted(entry["chapters"]),
            "mean_position_delta": round(entry["total_position_delta"] / entry["occurrences"], 4),
            "source_tokens": entry["source_tokens"],
            "target_tokens": entry["target_tokens"],
        }
        candidates.append(candidate)
        if entry["source_tokens"] <= 4 and entry["target_tokens"] <= 6:
            glossary_candidates.append(candidate)

    candidates.sort(key=lambda item: (-item["occurrences"], item["source_text"], item["target_text"]))
    glossary_candidates.sort(key=lambda item: (-item["occurrences"], item["source_text"], item["target_text"]))

    return {
        "work_slug": work_slug,
        "sampled_page_pairs": len(ocr_samples),
        "ocr_pages_processed": len(ocr_samples) * 2,
        "candidate_count": len(candidates),
        "glossary_candidate_count": len(glossary_candidates),
        "candidates": candidates[:500],
        "glossary_candidates": glossary_candidates[:200],
    }


def build_translation_memory_candidates(
    sample_pairs: list[dict],
    work_slug: str,
    ocr_runner=None,
) -> dict:
    ocr_samples = collect_ocr_sample_pairs(sample_pairs, ocr_runner=ocr_runner)
    return build_translation_memory_candidates_from_ocr_samples(ocr_samples, work_slug=work_slug)


def _summarize_side(ocr_samples: list[dict], key: str) -> dict:
    pages = len(ocr_samples)
    texts = [text for sample in ocr_samples for text in sample[key].get("texts", [])]
    total_regions = len(texts)
    total_chars = sum(len(_normalize_memory_text(text.get("text", ""))) for text in texts)
    type_distribution = Counter(text.get("tipo") or "desconhecido" for text in texts)
    source_distribution = Counter(text.get("ocr_source") or "unknown" for text in texts)

    return {
        "pages_processed": pages,
        "total_regions": total_regions,
        "mean_regions_per_page": round(total_regions / max(1, pages), 2),
        "mean_chars_per_region": round(total_chars / max(1, total_regions), 2),
        "text_type_distribution": dict(sorted(type_distribution.items())),
        "ocr_source_distribution": dict(sorted(source_distribution.items())),
    }


def build_textual_benchmark_profile_from_ocr_samples(
    ocr_samples: list[dict],
    work_slug: str,
) -> dict:
    length_ratios = []
    paired_region_count = 0
    for sample in ocr_samples:
        for pair in sample.get("paired_regions", []):
            source_text = _normalize_memory_text(pair["source"].get("text", ""))
            target_text = _normalize_memory_text(pair["target"].get("text", ""))
            if not (_is_valid_memory_text(source_text) and _is_valid_memory_text(target_text)):
                continue
            paired_region_count += 1
            length_ratios.append(len(target_text) / max(1, len(source_text)))

    return {
        "work_slug": work_slug,
        "sampled_page_pairs": len(ocr_samples),
        "ocr_pages_processed": len(ocr_samples) * 2,
        "en_stats": _summarize_side(ocr_samples, "en_result"),
        "pt_stats": _summarize_side(ocr_samples, "pt_result"),
        "paired_text_stats": {
            "paired_region_count": paired_region_count,
            "mean_translation_length_ratio": round(statistics.mean(length_ratios), 2) if length_ratios else 0.0,
        },
    }


def build_textual_benchmark_profile(
    sample_pairs: list[dict],
    work_slug: str,
    ocr_runner=None,
) -> dict:
    ocr_samples = collect_ocr_sample_pairs(sample_pairs, ocr_runner=ocr_runner)
    return build_textual_benchmark_profile_from_ocr_samples(ocr_samples, work_slug=work_slug)


def _materialize_sample_pairs(
    selected_pairs: list[dict],
    temp_dir: str | Path,
) -> list[dict]:
    materialized = []
    temp_root = Path(temp_dir)
    for index, pair in enumerate(selected_pairs, start=1):
        chapter = pair["chapter"]
        pair_dir = temp_root / f"chapter-{chapter:03d}" / f"pair-{index:03d}"
        en_path = _extract_cbz_page(pair["en_path"], pair["en_page"], pair_dir / "en.png")
        pt_path = _extract_cbz_page(pair["pt_path"], pair["pt_page"], pair_dir / "pt.png")
        materialized.append(
            {
                "chapter": chapter,
                "distance": pair["distance"],
                "en_image_path": en_path,
                "pt_image_path": pt_path,
                "en_page": pair["en_page"],
                "pt_page": pair["pt_page"],
            }
        )
    return materialized


def write_corpus_artifacts(
    pt_directory: str | Path,
    en_directory: str | Path,
    output_directory: str | Path,
    work_slug: str,
    max_ocr_page_pairs: int = 24,
    ocr_runner=None,
) -> dict:
    pt_entries = scan_pt_chapters(pt_directory)
    en_entries = scan_en_chapters(en_directory)
    pairs = pair_parallel_chapters(pt_entries, en_entries)
    manifest = build_manifest(work_slug, pt_entries, en_entries, pairs)
    quality = build_quality_profile(manifest)
    alignment = build_alignment_profile(manifest)
    work_profile = build_work_profile(manifest)
    visual_benchmark = build_visual_benchmark_profile(manifest)
    page_alignment = build_page_alignment_profile(manifest)

    selected_pairs = select_aligned_page_samples(
        manifest=manifest,
        page_alignment_profile=page_alignment,
        max_page_pairs=max_ocr_page_pairs,
    )

    if selected_pairs:
        with tempfile.TemporaryDirectory(prefix="traduzai-corpus-") as temp_dir:
            materialized_pairs = _materialize_sample_pairs(selected_pairs, temp_dir)
            ocr_samples = collect_ocr_sample_pairs(materialized_pairs, ocr_runner=ocr_runner)
    else:
        materialized_pairs = []
        ocr_samples = []

    textual_benchmark = build_textual_benchmark_profile_from_ocr_samples(
        ocr_samples,
        work_slug=work_slug,
    )
    translation_memory = build_translation_memory_candidates_from_ocr_samples(
        ocr_samples,
        work_slug=work_slug,
    )

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "manifest.json": manifest,
        "quality_profile.json": quality,
        "alignment_profile.json": alignment,
        "work_profile.json": work_profile,
        "visual_benchmark_profile.json": visual_benchmark,
        "page_alignment_profile.json": page_alignment,
        "textual_benchmark_profile.json": textual_benchmark,
        "translation_memory_candidates.json": translation_memory,
    }
    for filename, content in artifacts.items():
        (output_dir / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return {
        "manifest": manifest,
        "quality_profile": quality,
        "alignment_profile": alignment,
        "work_profile": work_profile,
        "visual_benchmark_profile": visual_benchmark,
        "page_alignment_profile": page_alignment,
        "textual_benchmark_profile": textual_benchmark,
        "translation_memory_candidates": translation_memory,
        "sampled_page_pairs": materialized_pairs,
        "output_directory": str(output_dir),
    }
