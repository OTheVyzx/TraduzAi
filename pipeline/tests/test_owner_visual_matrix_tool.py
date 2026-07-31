from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))


def _entry(work_id: str, work_dir: str, categories: list[str]) -> dict:
    return {
        "entry_id": f"entry_{work_id}",
        "work_id": work_id,
        "chapter_id": "chapter_fixture",
        "split": "calibration" if work_id == "work_a" else "holdout",
        "config_path": f"configs/{work_id}.json",
        "config_sha256": "0" * 64,
        "input_key": f"input_{work_id}",
        "expected_input_sha256": "1" * 64,
        "work_dir": work_dir,
        "categories": categories,
    }


def test_matrix_requires_multiple_works_and_visual_categories():
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_manifest

    with pytest.raises(MatrixContractError, match="three distinct works"):
        validate_manifest(
            {
                "entries": [
                    _entry("work_a", "run_a", ["white_balloon"]),
                    _entry("work_a", "run_b", ["burst"]),
                    _entry("work_b", "run_c", ["dark_panel"]),
                ]
            }
        )

    with pytest.raises(MatrixContractError, match="missing visual categories"):
        validate_manifest(
            {
                "entries": [
                    _entry("work_a", "run_a", ["white_balloon"]),
                    _entry("work_b", "run_b", ["burst"]),
                    _entry("work_c", "run_c", ["dark_panel"]),
                ]
            }
        )


def test_matrix_rejects_reused_output_directories():
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_manifest

    categories = [
        "white_balloon",
        "translucent_balloon",
        "burst",
        "dark_panel",
        "colored_card",
        "text_over_art",
        "table_ranking",
        "cross_tile_owner",
        "primary_ocr_omission",
    ]
    manifest = {
        "entries": [
            _entry("work_a", "same", categories),
            _entry("work_b", "same", categories),
            _entry("work_c", "run_c", categories),
        ]
    }
    with pytest.raises(MatrixContractError, match="unique work_dir"):
        validate_manifest(manifest)


def test_matrix_requires_fresh_hash_and_non_blocked_export_gate(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    work_dir = tmp_path / "run_a"
    translated = work_dir / "translated"
    translated.mkdir(parents=True)
    artifact = translated / "001.png"
    artifact.write_bytes(b"fresh-final-pixels")
    digest = sha256(artifact.read_bytes()).hexdigest()
    project = {
        "owner_graph_status": "verified",
        "qa": {
            "export_gate": {"status": "PASS", "issues": []},
            "final_pixel_reports": [
                {
                    "page_id": "page_001",
                    "artifact_path": str(artifact),
                    "persisted_sha256": digest,
                    "observer_available": True,
                    "observation_complete": True,
                    "coverage_complete": True,
                    "coverage_failures": [],
                    "contracts": {
                        "source_coverage_contract": "PASS",
                        "owner_graph_contract": "PASS",
                        "route_state_contract": "PASS",
                        "pixel_ownership_contract": "PASS",
                        "final_language_contract": "PASS",
                        "layout_legibility_contract": "PASS",
                        "residual_cleanup_contract": "PASS",
                        "protected_art_contract": "PASS",
                        "qa_integrity_contract": "PASS",
                    },
                    "issues": [],
                }
            ],
        },
    }
    (work_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")

    assert validate_entry_result(_entry("work_a", "run_a", []), tmp_path)["status"] == "PASS"
    artifact.write_bytes(b"mutated-after-qa")
    result = validate_entry_result(_entry("work_a", "run_a", []), tmp_path)
    assert result["status"] == "BLOCK"
    assert "final_artifact_hash_mismatch" in result["contracts"]

    project["qa"]["export_gate"]["status"] = "BLOCK"
    (work_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
    result = validate_entry_result(_entry("work_a", "run_a", []), tmp_path)
    assert "export_gate_blocked" in result["contracts"]


def test_matrix_report_groups_failures_by_contract_not_case_id():
    from tools.validate_owner_visual_matrix import group_failures_by_contract

    grouped = group_failures_by_contract(
        [
            {"entry_id": "case_a", "contracts": ["source_text_unowned"]},
            {"entry_id": "case_b", "contracts": ["source_text_unowned", "protected_art_damage"]},
        ]
    )

    assert grouped == {
        "protected_art_damage": ["case_b"],
        "source_text_unowned": ["case_a", "case_b"],
    }


def test_contact_sheet_category_selects_declared_page(tmp_path):
    from tools.validate_owner_visual_matrix import _selected_final_path

    translated = tmp_path / "translated"
    translated.mkdir()
    first = translated / "001.jpg"
    second = translated / "002.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    entry = {"category_pages": {"colored_card": 2}}

    assert _selected_final_path(entry, "colored_card", tmp_path) == second


def test_contact_sheet_category_selects_all_declared_pages(tmp_path):
    from tools.validate_owner_visual_matrix import _selected_final_paths

    translated = tmp_path / "translated"
    translated.mkdir()
    files = [translated / f"{index:03d}.jpg" for index in range(1, 4)]
    for path in files:
        path.write_bytes(path.name.encode("ascii"))

    assert _selected_final_paths(
        {"category_pages": {"burst": [1, 3]}}, "burst", tmp_path
    ) == [files[0], files[2]]


def test_contact_sheets_segment_tall_pages_at_readable_scale(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import _write_contact_sheets

    output_root = tmp_path / "matrix"
    work_dir = output_root / "entry_a"
    for folder in ("originals", "images", "translated"):
        (work_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 1000), "white").save(work_dir / folder / "001.png")
    entries = [
        {
            "entry_id": "entry_a",
            "work_dir": "entry_a",
            "categories": ["white_balloon"],
            "category_pages": {"white_balloon": 1},
        }
    ]

    sheets = _write_contact_sheets(entries, output_root)

    assert len(sheets["white_balloon"]) >= 2
    with Image.open(sheets["white_balloon"][0]) as sheet:
        assert sheet.width == 500
        assert sheet.height > 100


def test_runner_persists_captured_stdout_and_stderr(tmp_path):
    from tools.validate_owner_visual_matrix import _persist_runner_logs

    paths = _persist_runner_logs(
        tmp_path,
        "entry_a",
        stdout="pipeline progress",
        stderr="owner invariant failed",
    )

    assert Path(paths["stdout_path"]).read_text(encoding="utf-8") == "pipeline progress"
    assert Path(paths["stderr_path"]).read_text(encoding="utf-8") == "owner invariant failed"


def _passing_owner_project(artifact: Path) -> dict:
    digest = sha256(artifact.read_bytes()).hexdigest()
    return {
        "owner_graph_status": "verified",
        "paginas": [
            {
                "numero": 1,
                "page_id": "page_001",
                "text_layers": [
                    {
                        "id": "owner_a",
                        "owner_id": "owner_a",
                        "render_completed": True,
                        "route_action": "translate_inpaint_render",
                        "owner_render_quality": {
                            "status": "ok",
                            "source_scale_ratio": 1.0,
                            "outside_safe_pixels": 0,
                            "rendered_line_core_heights_px": [18],
                        },
                        "residual_cleanup_contract": {
                            "residual_verified": True,
                            "residual_score": 0.01,
                            "residual_threshold": 0.10,
                        },
                        "protected_art_contract": {
                            "protected_art_changed_pixels": 0,
                            "action_protected_overlap_pixels": 0,
                        },
                    }
                ],
            }
        ],
        "qa": {
            "export_gate": {"status": "PASS", "issues": []},
            "final_pixel_reports": [
                {
                    "page_id": "page_001",
                    "artifact_path": str(artifact),
                    "persisted_sha256": digest,
                    "observer_available": True,
                    "observation_complete": True,
                    "coverage_complete": True,
                    "coverage_failures": [],
                    "contracts": {
                        "source_coverage_contract": "PASS",
                        "owner_graph_contract": "PASS",
                        "route_state_contract": "PASS",
                        "pixel_ownership_contract": "PASS",
                        "final_language_contract": "PASS",
                        "layout_legibility_contract": "PASS",
                        "residual_cleanup_contract": "PASS",
                        "protected_art_contract": "PASS",
                        "qa_integrity_contract": "PASS",
                    },
                    "issues": [],
                }
            ],
        },
    }


def _write_owner_project(tmp_path: Path, project: dict) -> tuple[dict, Path]:
    work_dir = tmp_path / "run_a"
    work_dir.mkdir(exist_ok=True)
    (work_dir / "project.json").write_text(json.dumps(project), encoding="utf-8")
    return _entry("work_a", "run_a", []), work_dir


def test_matrix_blocks_missing_owner_render_quality_contract(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    project["paginas"][0]["text_layers"][0].pop("owner_render_quality")
    entry, _ = _write_owner_project(tmp_path, project)

    result = validate_entry_result(entry, tmp_path)
    assert "missing_owner_render_quality_contract" in result["contracts"]


def test_matrix_blocks_under_source_scale_owner(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    quality = project["paginas"][0]["text_layers"][0]["owner_render_quality"]
    quality.update({"status": "under_source_scale", "source_scale_ratio": 0.5})
    entry, _ = _write_owner_project(tmp_path, project)

    assert "under_source_scale" in validate_entry_result(entry, tmp_path)["contracts"]


def test_matrix_blocks_incomplete_final_ocr_coverage(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    report = project["qa"]["final_pixel_reports"][0]
    report.update({"coverage_complete": False, "coverage_failures": ["detected_blocks_without_ocr"]})
    entry, _ = _write_owner_project(tmp_path, project)

    result = validate_entry_result(entry, tmp_path)
    assert "detected_blocks_without_ocr" in result["contracts"]


def test_matrix_blocks_unverified_residual(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    project["paginas"][0]["text_layers"][0]["residual_cleanup_contract"]["residual_verified"] = False
    entry, _ = _write_owner_project(tmp_path, project)

    assert "unverified_owner_residual" in validate_entry_result(entry, tmp_path)["contracts"]


def test_matrix_groups_failures_by_contract():
    from tools.validate_owner_visual_matrix import group_failures_by_contract

    assert group_failures_by_contract(
        [
            {"entry_id": "a", "contracts": ["layout_legibility_contract"]},
            {"entry_id": "b", "contracts": ["layout_legibility_contract", "qa_integrity_contract"]},
        ]
    ) == {
        "layout_legibility_contract": ["a", "b"],
        "qa_integrity_contract": ["b"],
    }


def test_matrix_requires_calibration_and_holdout_entries():
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_manifest

    categories = sorted(REQUIRED_VISUAL_CATEGORIES_FOR_TEST)
    entries = [_entry(f"work_{suffix}", f"run_{suffix}", categories) for suffix in ("a", "b", "c")]
    for entry in entries:
        entry["split"] = "holdout"
    with pytest.raises(MatrixContractError, match="calibration and holdout"):
        validate_manifest({"entries": entries})


REQUIRED_VISUAL_CATEGORIES_FOR_TEST = {
    "white_balloon", "translucent_balloon", "burst", "dark_panel", "colored_card",
    "text_over_art", "table_ranking", "cross_tile_owner", "primary_ocr_omission",
}


def _canonical_json_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def test_matrix_resolves_versioned_config_and_input_key_with_matching_hash(tmp_path, monkeypatch):
    from tools.validate_owner_visual_matrix import resolve_entry_runtime, _sha256_input

    fixture = tmp_path / "fixture"
    (fixture / "configs").mkdir(parents=True)
    config = fixture / "configs" / "case.json"
    config.write_text(json.dumps({"input_key": "case_input"}), encoding="utf-8")
    source = tmp_path / "source.cbz"
    source.write_bytes(b"source")
    input_hash = _sha256_input(source)
    inputs = {
        "inputs": {
            "case_input": {
                "environment_variable": "TRADUZAI_MATRIX_CASE",
                "input_type": "file",
                "expected_sha256": input_hash,
            }
        }
    }
    monkeypatch.setenv("TRADUZAI_MATRIX_CASE", str(source))
    entry = {
        "config_path": "configs/case.json",
        "config_sha256": _canonical_json_hash(config),
        "input_key": "case_input",
        "expected_input_sha256": input_hash,
    }

    resolved = resolve_entry_runtime(entry, fixture / "matrix.json", inputs)
    assert resolved["config_path"] == config.resolve()
    assert resolved["source_path"] == source.resolve()
    assert resolved["input_sha256"] == input_hash


def test_matrix_rejects_missing_or_hash_mismatched_input(tmp_path, monkeypatch):
    from tools.validate_owner_visual_matrix import MatrixContractError, resolve_entry_runtime

    fixture = tmp_path / "fixture"
    (fixture / "configs").mkdir(parents=True)
    config = fixture / "configs" / "case.json"
    config.write_text(json.dumps({"input_key": "case_input"}), encoding="utf-8")
    entry = {
        "config_path": "configs/case.json",
        "config_sha256": _canonical_json_hash(config),
        "input_key": "case_input",
        "expected_input_sha256": "a" * 64,
    }
    inputs = {"inputs": {"case_input": {"environment_variable": "TRADUZAI_MATRIX_CASE", "input_type": "file", "expected_sha256": "a" * 64}}}
    monkeypatch.setenv("TRADUZAI_MATRIX_CASE", str(tmp_path / "missing.cbz"))
    with pytest.raises(MatrixContractError, match="input missing"):
        resolve_entry_runtime(entry, fixture / "matrix.json", inputs)
    source = tmp_path / "source.cbz"
    source.write_bytes(b"wrong")
    monkeypatch.setenv("TRADUZAI_MATRIX_CASE", str(source))
    with pytest.raises(MatrixContractError, match="hash mismatch"):
        resolve_entry_runtime(entry, fixture / "matrix.json", inputs)


def test_inspection_manifest_requires_artifact_hash_scale_and_verdict(tmp_path):
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_inspection_manifest

    artifact = tmp_path / "sheet.png"
    artifact.write_bytes(b"sheet")
    template = {"artifacts": [{"artifact_path": "sheet.png", "sha256": sha256(b"sheet").hexdigest(), "category": "burst", "segment": "entry:1"}]}
    incomplete = {"inspections": [{"artifact_path": "sheet.png"}]}
    with pytest.raises(MatrixContractError, match="inspection fields"):
        validate_inspection_manifest(template, incomplete, tmp_path)


def test_report_rejects_unverified_inspection_claims(tmp_path):
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_inspection_manifest

    artifact = tmp_path / "sheet.png"
    artifact.write_bytes(b"sheet")
    template = {"artifacts": [{"artifact_path": "sheet.png", "sha256": sha256(b"sheet").hexdigest(), "category": "burst", "segment": "entry:1"}]}
    manifest = {"inspections": [{"artifact_path": "sheet.png", "sha256": "0" * 64, "scale": "native", "timestamp": "2026-07-30T12:00:00Z", "category": "burst", "owner_or_segment": "entry:1", "verdict": "PASS", "note": "clean"}]}
    with pytest.raises(MatrixContractError, match="inspection hash"):
        validate_inspection_manifest(template, manifest, tmp_path)


def test_report_lists_only_artifacts_actually_inspected(tmp_path):
    from tools.validate_owner_visual_matrix import _write_report

    checked = tmp_path / "checked.png"
    unchecked = tmp_path / "unchecked.png"
    checked.write_bytes(b"a")
    unchecked.write_bytes(b"b")
    report = tmp_path / "report.md"
    _write_report(
        report,
        [{"entry_id": "a", "categories": ["burst"], "page_count": 1, "export_gate": "PASS", "status": "PASS", "contracts": []}],
        [],
        {"burst": [str(checked), str(unchecked)]},
        inspected=[{"artifact_path": str(checked), "sha256": sha256(b"a").hexdigest(), "verdict": "PASS", "note": "native check"}],
    )
    visual_section = report.read_text(encoding="utf-8").split("## Visually inspected", 1)[1]
    assert str(checked) in visual_section
    assert str(unchecked) not in visual_section
