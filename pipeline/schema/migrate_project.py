from __future__ import annotations

from copy import deepcopy
from typing import Any

from .project_schema_v12 import (
    SCHEMA_VERSION,
    build_empty_project_v12,
    build_empty_region_v12,
    with_recomputed_qa_summary,
)


def _merge_missing(target: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    for key, value in defaults.items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        if isinstance(target[key], dict) and isinstance(value, dict):
            _merge_missing(target[key], value)
    return target


def _as_bbox(value: Any) -> list[int]:
    if isinstance(value, list) and len(value) >= 4:
        return [int(item) if isinstance(item, (int, float)) else 0 for item in value[:4]]
    return [0, 0, 0, 0]


def _region_type(tipo: Any) -> str:
    normalized = str(tipo or "").strip().lower()
    mapping = {
        "fala": "speech_balloon",
        "speech": "speech_balloon",
        "speech_balloon": "speech_balloon",
        "narracao": "caption",
        "narração": "caption",
        "caption": "caption",
        "sfx": "sfx",
        "onomatopeia": "sfx",
        "background_text": "background_text",
        "fundo": "background_text",
    }
    return mapping.get(normalized, "unknown")


def _text_value(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _region_from_text(text_item: dict[str, Any], page_num: int, region_index: int) -> dict[str, Any]:
    region = build_empty_region_v12(page=page_num, index=region_index)
    raw_ocr = _text_value(text_item, "original", "texto", "text", "raw_ocr")
    translated = _text_value(text_item, "translated", "traduzido", "translation")
    confidence = text_item.get("ocr_confidence", text_item.get("confianca_ocr", text_item.get("confidence", 0.0)))

    region["region_id"] = str(text_item.get("id") or region["region_id"])
    if not region["region_id"].startswith("p"):
        region["region_id"] = f"p{page_num:03}_r{region_index:03}"
    region["bbox"] = _as_bbox(text_item.get("bbox") or text_item.get("source_bbox") or text_item.get("layout_bbox"))
    region["reading_order"] = int(text_item.get("order", region_index - 1) or 0)
    region["region_type"] = _region_type(text_item.get("tipo") or text_item.get("region_type"))
    region["raw_ocr"] = raw_ocr
    region["normalized_ocr"] = _text_value(text_item, "normalized_ocr") or raw_ocr
    region["ocr_confidence"] = float(confidence or 0.0)
    region["translation"]["text"] = translated
    region["translation"]["used_glossary"] = deepcopy(text_item.get("glossary_hits", []))
    region["entities"] = deepcopy(text_item.get("entities", []))
    region["qa_flags"] = list(text_item.get("qa_flags", []))
    region["mask"]["bbox"] = deepcopy(text_item.get("balloon_bbox") or region["bbox"])
    region["layout"]["font"] = str((text_item.get("style") or text_item.get("estilo") or {}).get("fonte", ""))
    region["layout"]["font_size"] = int((text_item.get("style") or text_item.get("estilo") or {}).get("tamanho", 0) or 0)
    return region


def _page_regions(page: dict[str, Any], page_num: int) -> list[dict[str, Any]]:
    text_items = page.get("text_layers")
    if not isinstance(text_items, list):
        text_items = page.get("textos", [])
    if not isinstance(text_items, list):
        return []
    return [
        _region_from_text(item, page_num, index)
        for index, item in enumerate(text_items, start=1)
        if isinstance(item, dict)
    ]


def _flags_from_regions(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for page in pages:
        for region in page.get("regions", []):
            for reason in region.get("qa_flags", []):
                flags.append(
                    {
                        "page": page["page"],
                        "region_id": region["region_id"],
                        "severity": "medium",
                        "reason": str(reason),
                    }
                )
    return flags


def migrate_project_to_v12(project: dict[str, Any], *, input_path: str = "", mode: str = "mock") -> dict[str, Any]:
    current = deepcopy(project)
    if current.get("schema_version") == SCHEMA_VERSION:
        defaults = build_empty_project_v12(
            input_path=current.get("source", {}).get("input_path", input_path),
            page_count=current.get("source", {}).get("page_count", len(current.get("pages", []))),
            mode=current.get("run", {}).get("mode", mode),
        )
        migrated = _merge_missing(current, defaults)
        migrated.setdefault("legacy", {}).setdefault("paginas", current.get("paginas", []))
        return with_recomputed_qa_summary(migrated)

    legacy_pages = current.get("paginas", [])
    if not isinstance(legacy_pages, list):
        legacy_pages = []

    migrated = build_empty_project_v12(input_path=input_path, page_count=len(legacy_pages), mode=mode)
    migrated["legacy"]["paginas"] = deepcopy(legacy_pages)
    migrated["work_context"]["title"] = current.get("obra")
    migrated["work_context"]["selected"] = bool(current.get("obra"))
    migrated["work_context"]["context_loaded"] = bool(current.get("contexto"))
    glossary = (current.get("contexto") or {}).get("glossario") if isinstance(current.get("contexto"), dict) else None
    if isinstance(glossary, dict):
        migrated["work_context"]["glossary_loaded"] = bool(glossary)
        migrated["work_context"]["glossary_entries_count"] = len(glossary)

    pages: list[dict[str, Any]] = []
    for page_index, legacy_page in enumerate(legacy_pages, start=1):
        if not isinstance(legacy_page, dict):
            continue
        page_num = int(legacy_page.get("numero", page_index) or page_index)
        pages.append(
            {
                "page": page_num,
                "source_path": legacy_page.get("arquivo_original"),
                "rendered_path": legacy_page.get("arquivo_traduzido"),
                "width": legacy_page.get("width"),
                "height": legacy_page.get("height"),
                "regions": _page_regions(legacy_page, page_num),
            }
        )

    migrated["pages"] = pages
    migrated["qa"]["flags"] = _flags_from_regions(pages)
    return with_recomputed_qa_summary(migrated)

