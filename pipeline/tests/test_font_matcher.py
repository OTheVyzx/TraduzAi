"""Rendered-source-shape font matcher contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent
FONTS_DIR = REPO_ROOT / "fonts"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.font_matcher import (  # noqa: E402
    FontCatalogEntry,
    FontShapeMatcher,
    font_match_cache_key,
    load_font_catalog,
    render_source_text_mask,
)


def _catalog() -> tuple[FontCatalogEntry, ...]:
    return load_font_catalog(FONTS_DIR, FONTS_DIR / "font-map.json")


def test_source_text_rerender_ranks_exact_font_first() -> None:
    source_text = "SOURCE BODY 123"
    profile = {"rotation_deg": 0.0, "slant_tangent": 0.0, "width_scale": 1.0}
    source = render_source_text_mask(
        source_text,
        FONTS_DIR / "ComicNeue-Bold.ttf",
        profile=profile,
    )

    result = FontShapeMatcher(_catalog()).match(
        source,
        source_text=source_text,
        translated_text="CORPO TRADUZIDO",
        profile=profile,
        semantic_role="dialogue_body",
    )

    assert result.value == "ComicNeue-Bold.ttf"
    assert result.status == "exact"
    assert result.top_k[0]["font_name"] == "ComicNeue-Bold.ttf"
    assert result.confidence >= 0.8


def test_matcher_abstains_when_top_two_margin_is_ambiguous() -> None:
    font_path = FONTS_DIR / "ComicNeue-Bold.ttf"
    catalog = (
        FontCatalogEntry("alias-a.ttf", font_path, ("dialogue_body",), "bundled-project-asset"),
        FontCatalogEntry("alias-b.ttf", font_path, ("dialogue_body",), "bundled-project-asset"),
    )
    source = render_source_text_mask("AMBIGUOUS", font_path, profile={})

    result = FontShapeMatcher(catalog).match(
        source,
        source_text="AMBIGUOUS",
        profile={},
        semantic_role="dialogue_body",
    )

    assert result.value == "unknown"
    assert result.selected_font is None
    assert result.status == "unknown"
    assert result.abstention_reason == "insufficient_margin"


def test_matcher_uses_frozen_source_text_not_translation() -> None:
    catalog = _catalog()
    source_text = "FROZEN SOURCE"
    profile = {"width_scale": 1.0}
    source = render_source_text_mask(
        source_text, FONTS_DIR / "KOMIKAX_.ttf", profile=profile
    )
    matcher = FontShapeMatcher(catalog)

    first = matcher.match(
        source,
        source_text=source_text,
        translated_text="PRIMEIRA TRADUCAO",
        profile=profile,
        semantic_role="sfx",
    )
    second = matcher.match(
        source,
        source_text=source_text,
        translated_text="OUTRA TRADUCAO TOTALMENTE DIFERENTE",
        profile=profile,
        semantic_role="sfx",
    )

    assert first.value == second.value == "KOMIKAX_.ttf"
    assert first.cache_key == second.cache_key


@pytest.mark.parametrize("width_scale", [0.62, 1.28])
def test_matcher_handles_condensed_and_expanded_width(width_scale: float) -> None:
    profile = {"width_scale": width_scale, "rotation_deg": 0.0, "slant_tangent": 0.0}
    font = FONTS_DIR / "LeagueGothic-Regular-VariableFont_wdth.ttf"
    source = render_source_text_mask("WIDTH PROFILE", font, profile=profile)

    result = FontShapeMatcher(_catalog()).match(
        source,
        source_text="WIDTH PROFILE",
        profile=profile,
        semantic_role="system_card",
    )

    assert result.value == font.name
    assert result.status == "exact"


def test_font_catalog_contains_only_existing_licensed_files() -> None:
    raw = json.loads((FONTS_DIR / "font-map.json").read_text(encoding="utf-8"))
    catalog = _catalog()

    assert len(catalog) == len(raw["available"])
    assert all(entry.path.is_file() for entry in catalog)
    assert all(entry.license_status for entry in catalog)
    assert all(entry.roles for entry in catalog)


def test_matcher_cache_key_includes_catalog_and_profile_hashes() -> None:
    catalog = _catalog()
    base = font_match_cache_key(catalog, "CACHE SOURCE", {"width_scale": 1.0})
    changed_profile = font_match_cache_key(
        catalog, "CACHE SOURCE", {"width_scale": 0.7}
    )
    changed_catalog = font_match_cache_key(
        catalog[:-1], "CACHE SOURCE", {"width_scale": 1.0}
    )

    assert base != changed_profile
    assert base != changed_catalog


def test_matcher_records_catalog_version_and_is_catalog_order_invariant() -> None:
    catalog = _catalog()
    source = render_source_text_mask(
        "ORDER",
        FONTS_DIR / "ComicNeue-Bold.ttf",
        profile={},
    )

    first = FontShapeMatcher(catalog).match(
        source,
        source_text="ORDER",
        profile={},
        semantic_role="dialogue_body",
    )
    second = FontShapeMatcher(tuple(reversed(catalog))).match(
        source,
        source_text="ORDER",
        profile={},
        semantic_role="dialogue_body",
    )

    assert first == second
    assert len(first.catalog_version) == 64
    assert first.normalization["canvas"] == [192, 512]


def test_benchmark_shortlist_still_publishes_real_ranked_top_k() -> None:
    source_text = "MASK BACKED SOURCE"
    profile = {"width_scale": 1.0, "rotation_deg": 0.0}
    source = render_source_text_mask(
        source_text, FONTS_DIR / "ComicNeue-Bold.ttf", profile=profile
    )

    result = FontShapeMatcher(_catalog()).match(
        source,
        source_text=source_text,
        profile=profile,
        semantic_role="dialogue_body",
        shortlist=("ComicNeue-Bold.ttf", "LeagueGothic-Regular-VariableFont_wdth.ttf"),
    )

    assert [entry["font_name"] for entry in result.top_k] == [
        "ComicNeue-Bold.ttf",
        "LeagueGothic-Regular-VariableFont_wdth.ttf",
    ]
