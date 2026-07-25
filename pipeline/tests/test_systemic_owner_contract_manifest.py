import json
from pathlib import Path


MANIFEST_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "systemic_visual_contracts"
    / "owner_cases.json"
)

REQUIRED_CATEGORIES = {
    "overlapping_tiles_single_body",
    "card_semantic_roles",
    "connected_layout_regions",
    "repeated_text_distinct_geometry",
    "protected_art_contact",
    "primary_ocr_omission_recall",
}

REQUIRED_CASE_FIELDS = {
    "case_id",
    "category",
    "page_size",
    "source_components",
    "observations",
    "tiles",
    "protected_art",
    "expected",
}


def _load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"missing systemic owner fixture: {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_covers_every_required_systemic_category():
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    categories = {case["category"] for case in manifest["cases"]}
    assert categories == REQUIRED_CATEGORIES


def test_manifest_cases_are_generic_and_forbid_production_exceptions():
    cases = _load_manifest()["cases"]

    for case in cases:
        assert REQUIRED_CASE_FIELDS <= case.keys()
        assert "work" not in case
        assert "chapter" not in case
        assert "page_number" not in case
        assert case["expected"]["production_exception_count"] == 0


def test_manifest_identifiers_are_unique_within_each_case():
    cases = _load_manifest()["cases"]

    assert len({case["case_id"] for case in cases}) == len(cases)
    for case in cases:
        component_ids = [item["component_id"] for item in case["source_components"]]
        observation_ids = [item["observation_id"] for item in case["observations"]]
        tile_ids = [item["tile_id"] for item in case["tiles"]]
        assert len(component_ids) == len(set(component_ids))
        assert len(observation_ids) == len(set(observation_ids))
        assert len(tile_ids) == len(set(tile_ids))


def test_manifest_exercises_one_two_and_four_tile_partitions():
    tile_counts = {len(case["tiles"]) for case in _load_manifest()["cases"]}

    assert {1, 2, 4} <= tile_counts
