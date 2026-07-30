import hashlib
import json
from pathlib import Path


MANIFEST_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "systemic_visual_contracts"
    / "owner_cases.json"
)
MATRIX_ROOT = Path(__file__).parent / "fixtures" / "owner_visual_matrix"
FUNCTIONAL_MATRIX_PATH = MATRIX_ROOT / "functional_matrix.json"
INPUTS_PATH = MATRIX_ROOT / "inputs.json"

REQUIRED_CATEGORIES = {
    "overlapping_tiles_single_body",
    "card_semantic_roles",
    "connected_layout_regions",
    "repeated_text_distinct_geometry",
    "protected_art_contact",
    "primary_ocr_omission_recall",
    "tight_truncation_vs_complete_numeric_body",
    "cross_band_background_continuity",
    "proportional_layout_underfill",
    "incomplete_final_observation",
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


def _load_json(path: Path) -> dict:
    assert path.exists(), f"missing versioned owner visual fixture: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    canonical = json.dumps(
        json.loads(path.read_text(encoding="utf-8")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def test_complete_numeric_case_contains_competing_full_and_truncated_evidence():
    case = next(
        item
        for item in _load_manifest()["cases"]
        if item["category"] == "tight_truncation_vs_complete_numeric_body"
    )

    assert {item["coverage_kind"] for item in case["observations"]} == {
        "complete",
        "truncated",
    }
    assert sum(
        item["coverage_kind"] == "complete" for item in case["observations"]
    ) >= 2
    assert case["expected"]["selected_coverage_kind"] == "complete"
    assert case["expected"]["production_exception_count"] == 0


def test_functional_matrix_is_versioned_and_resolves_only_versioned_configs():
    matrix = _load_json(FUNCTIONAL_MATRIX_PATH)
    entries = matrix["entries"]

    assert matrix["schema_version"] == 1
    assert entries
    assert {entry["split"] for entry in entries} == {"calibration", "holdout"}
    assert len({entry["entry_id"] for entry in entries}) == len(entries)
    for entry in entries:
        assert {
            "entry_id",
            "work_id",
            "chapter_id",
            "split",
            "config_path",
            "config_sha256",
            "input_key",
            "expected_input_sha256",
            "categories",
            "category_pages",
        } <= entry.keys()
        config_path = (MATRIX_ROOT / entry["config_path"]).resolve()
        assert config_path.is_relative_to(MATRIX_ROOT.resolve())
        assert config_path.is_file()
        assert _sha256(config_path) == entry["config_sha256"]


def test_functional_matrix_inputs_are_portable_and_hash_pinned():
    matrix = _load_json(FUNCTIONAL_MATRIX_PATH)
    inputs = _load_json(INPUTS_PATH)
    input_rows = inputs["inputs"]

    assert inputs["schema_version"] == 1
    assert set(input_rows) == {entry["input_key"] for entry in matrix["entries"]}
    for entry in matrix["entries"]:
        row = input_rows[entry["input_key"]]
        assert set(row) == {"environment_variable", "input_type", "expected_sha256"}
        assert row["environment_variable"].startswith("TRADUZAI_MATRIX_")
        assert row["input_type"] in {"directory", "file"}
        assert len(row["expected_sha256"]) == 64
        int(row["expected_sha256"], 16)
        assert entry["expected_input_sha256"] == row["expected_sha256"]


def test_versioned_matrix_and_configs_contain_no_runtime_or_absolute_paths():
    fixture_paths = [FUNCTIONAL_MATRIX_PATH, INPUTS_PATH]
    fixture_paths.extend(MATRIX_ROOT.glob("configs/*.json"))

    for path in fixture_paths:
        raw = path.read_text(encoding="utf-8")
        assert ".codex-tmp" not in raw
        assert "N:\\" not in raw
        assert "source_path" not in raw
        if path.parent.name == "configs":
            config = json.loads(raw)
            assert config["input_key"]
            assert "work_dir" not in config
            assert "models_dir" not in config
