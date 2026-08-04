from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.hash_contract import canonical_source_tree_sha256  # noqa: E402


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "english_owner_recovery"
REQUIRED_CATEGORIES = {
    "cross_page_ocr_leak",
    "no_band_component",
    "unassociated_full_page_observation",
    "semantic_container_missing",
    "atomic_cleanup_rollback",
    "mixed_language_overlay",
    "final_target_missing",
    "qa_duplicate_false_positive",
    "burst_container",
    "card_table_multirole",
    "text_over_protected_art",
    "cross_tile_owner",
}


def _load_manifest() -> dict:
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _load_recipes() -> dict:
    return json.loads((FIXTURE_ROOT / "recipes.json").read_text(encoding="utf-8"))["recipes"]


def _ordered_images(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.png"), key=lambda path: path.relative_to(root).as_posix()))


def _write_small_external_input(tmp_path: Path) -> Path:
    source = tmp_path / "portable-source"
    source.mkdir()
    Image.new("RGB", (4, 3), (12, 34, 56)).save(source / "001.png")
    Image.new("RGB", (4, 3), (78, 90, 123)).save(source / "002.png")
    return source


def _write_nested_source_tree(tmp_path: Path) -> Path:
    source = tmp_path / "nested-source"
    (source / "chapter").mkdir(parents=True)
    Image.new("RGB", (3, 2), (1, 2, 3)).save(source / "chapter" / "001.png")
    Image.new("RGB", (3, 2), (4, 5, 6)).save(source / "002.png")
    return source


def _mutate_one_source_pixel(path: Path) -> None:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.putpixel((0, 0), (233, 17, 91))
    image.save(path)


def recompute_manifest_input_tree_sha256(input_key: str) -> str:
    source_value = os.environ.get(input_key)
    if not source_value:
        raise RuntimeError(f"missing source input environment variable: {input_key}")
    source = Path(source_value).resolve(strict=True)
    return canonical_source_tree_sha256(_ordered_images(source), source)


def test_manifest_covers_every_universal_failure_category():
    manifest = _load_manifest()
    recipes = _load_recipes()
    assert {case["category"] for case in manifest["cases"]} >= REQUIRED_CATEGORIES
    assert {case["recipe_id"] for case in manifest["cases"]} <= set(recipes)
    assert all("work_title" not in case for case in manifest["cases"])
    assert all("page_number_rule" not in case for case in manifest["cases"])


def test_every_synthetic_case_requires_both_modes_source_absence_and_target_presence():
    manifest = _load_manifest()
    assert {case["cohort"] for case in manifest["cases"]} == {"calibration", "holdout"}
    for case in manifest["cases"]:
        assert case["style_copy_modes"] == ["off", "render"]
        assert case["expected"] == {
            "english_dialogue_residual_count": 0,
            "source_absent": True,
            "target_present": True,
        }


def test_recipes_are_generic_declarative_and_contain_no_external_paths():
    recipes = _load_recipes()
    serialized = json.dumps(recipes, ensure_ascii=False)
    for recipe_id, recipe in recipes.items():
        assert recipe_id.startswith("generic_")
        assert {"canvas", "container", "glyph_layers", "language", "induced_defect"} <= set(recipe)
    assert ".codex-tmp" not in serialized
    assert not re.search(r"[A-Za-z]:[\\/]", serialized)


def test_real_regression_uses_hashed_external_inputs_not_production_rules():
    manifest = _load_manifest()
    real = manifest["real_regressions"]["mitch_items_ch39"]
    assert real["input_key"] == "TRADUZAI_MATRIX_MITCH39_SOURCE"
    assert set(real["sentinel_pages"]) == {10, 11, 19, 21, 27, 28, 30, 34, 36, 39}
    assert re.fullmatch(r"[0-9a-f]{64}", real["source_tree_sha256"])
    assert "source_path" not in real
    assert real["style_copy_modes"] == ["off", "render"]
    assert real["expected"]["target_present"] is True
    assert all(
        case["expected"]["english_dialogue_residual_count"] == 0
        for case in manifest["cases"]
    )


def test_manifest_input_tree_hash_recomputation_is_portable(tmp_path, monkeypatch):
    source = _write_small_external_input(tmp_path)
    monkeypatch.setenv("TRADUZAI_TEST_SOURCE", str(source))
    expected = canonical_source_tree_sha256(_ordered_images(source), source)
    assert recompute_manifest_input_tree_sha256("TRADUZAI_TEST_SOURCE") == expected


def test_canonical_source_tree_hash_binds_order_relative_paths_file_and_pixel_hashes(tmp_path):
    root = _write_nested_source_tree(tmp_path)
    ordered = _ordered_images(root)
    first = canonical_source_tree_sha256(ordered, root)
    assert first == canonical_source_tree_sha256(ordered, root)
    assert first != canonical_source_tree_sha256(tuple(reversed(ordered)), root)
    _mutate_one_source_pixel(ordered[0])
    assert first != canonical_source_tree_sha256(ordered, root)


@pytest.mark.integration
def test_real_regression_source_tree_hash_matches_runtime_input(monkeypatch):
    source_value = os.environ.get("TRADUZAI_MATRIX_MITCH39_SOURCE")
    if not source_value:
        pytest.skip("TRADUZAI_MATRIX_MITCH39_SOURCE is required for the real corpus integration")
    source = Path(source_value).resolve(strict=True)
    monkeypatch.setenv("TRADUZAI_MATRIX_MITCH39_SOURCE", str(source))
    real = _load_manifest()["real_regressions"]["mitch_items_ch39"]
    assert recompute_manifest_input_tree_sha256(real["input_key"]) == real["source_tree_sha256"]
