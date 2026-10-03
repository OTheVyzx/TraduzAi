"""Bridge the verified Vision pre-OCR cache into Consumer Fast.

The cache is advisory. A miss or invalid artifact always runs the real visual
stages; it never supplies partially verified page data.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from shutil import copy2
from typing import Any


def _enabled() -> bool:
    return os.getenv("TRADUZAI_EXPERIMENTAL_VISION_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}


def select_cached_pages(cache_root: Path, source_manifest: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not _enabled():
        return {"status": "disabled", "reason": "experimental_flag_off", "pages": {},
                "visual_config_sha256": None, "lineage": None}
    from vision_runtime.cache_key import visual_config_sha256
    from vision_runtime.cache_reader import cached_content_lineage, read_verified_chapter_analysis

    visual_hash = visual_config_sha256(config)
    result = read_verified_chapter_analysis(
        cache_root,
        source_tree_sha256=source_manifest.source_tree_sha256,
        page_source_sha256s={page.page_id: page.source_file_sha256 for page in source_manifest.pages},
        visual_config_sha256=visual_hash,
    )
    if result["status"] == "hit":
        lineage = cached_content_lineage(result["pages"])
        if lineage is None:
            return {"status": "miss", "reason": "cached_page_lineage_inconsistent", "pages": {},
                    "visual_config_sha256": visual_hash, "lineage": None}
        return {**result, "visual_config_sha256": visual_hash, "lineage": lineage}
    return {**result, "visual_config_sha256": visual_hash, "lineage": None}


def publish_cached_pages(
    cache_root: Path,
    execution_artifact_root: Path,
    output_pages: list[Any],
    config: dict[str, Any],
    source_manifest: Any,
) -> dict[str, Any]:
    """Publish a complete page set only after both pre-OCR stage files exist."""
    if not _enabled():
        return {"status": "not_written", "reason": "experimental_flag_off"}
    from vision_runtime.page_record import write_chapter_analysis_records

    source_stage = execution_artifact_root / "vision" / "pre_ocr"
    destination_stage = cache_root / "vision" / "pre_ocr"
    page_ids = [page.page_id for page in source_manifest.pages]
    if len(page_ids) != len(output_pages) or len(set(page_ids)) != len(page_ids):
        return {"status": "not_written", "reason": "page_set_mismatch"}
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", page_id) for page_id in page_ids):
        return {"status": "not_written", "reason": "unsafe_page_id"}
    pairs = [(source_stage / f"{page_id}.{suffix}", destination_stage / f"{page_id}.{suffix}")
             for page_id in page_ids for suffix in ("json", "npz")]
    missing = [source.name for source, _ in pairs if not source.is_file()]
    if missing:
        return {"status": "not_written", "reason": "pre_ocr_stage_missing", "missing": missing}
    destination_stage.mkdir(parents=True, exist_ok=True)
    for source, destination in pairs:
        copy2(source, destination)
    receipt = write_chapter_analysis_records(
        cache_root, output_pages, config,
        source_manifest_sha256=source_manifest.sha256,
        source_tree_sha256=source_manifest.source_tree_sha256,
    )
    return {"status": "written", **receipt}
