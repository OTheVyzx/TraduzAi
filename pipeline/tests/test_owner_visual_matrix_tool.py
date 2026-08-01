from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import subprocess

import pytest


PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))


def test_direct_matrix_cli_can_load_owner_target_tool():
    result = subprocess.run(
        [sys.executable, str(PIPELINE / "tools" / "validate_owner_visual_matrix.py"), "--help"],
        cwd=PIPELINE.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
        "targets": [
            {
                "page_id": "page_001",
                "owner_id": f"owner_{index}",
                "component_ids": [f"component_{index}"],
                "category": category,
                "split": "calibration" if work_id == "work_a" else "holdout",
                "minimum_count": 1,
                "source_crop": {"bbox_page": [0, 0, 1, 1], "width": 1, "height": 1, "sha256": "2" * 64},
            }
            for index, category in enumerate(categories)
        ],
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
    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "paginas": [
                    {"numero": 1, "image_layers": {"rendered": {"path": "translated/001.jpg"}}},
                    {"numero": 2, "image_layers": {"rendered": {"path": "translated/002.jpg"}}},
                ],
                "page_owner_graphs": [
                    {"page_id": "page_002", "owners": [{"owner_id": "owner_b", "component_ids": ["component_b"]}]}
                ],
            }
        ),
        encoding="utf-8",
    )
    entry = {"targets": [{"page_id": "page_002", "owner_id": "owner_b", "component_ids": ["component_b"], "category": "colored_card"}]}

    assert _selected_final_path(entry, "colored_card", tmp_path) == second


def test_contact_sheet_category_selects_all_declared_pages(tmp_path):
    from tools.validate_owner_visual_matrix import _selected_final_paths

    translated = tmp_path / "translated"
    translated.mkdir()
    files = [translated / f"{index:03d}.jpg" for index in range(1, 4)]
    for path in files:
        path.write_bytes(path.name.encode("ascii"))

    (tmp_path / "project.json").write_text(
        json.dumps(
            {
                "paginas": [
                    {"numero": index, "image_layers": {"rendered": {"path": f"translated/{index:03d}.jpg"}}}
                    for index in range(1, 4)
                ],
                "page_owner_graphs": [
                    {"page_id": page_id, "owners": [{"owner_id": owner_id, "component_ids": [component_id]}]}
                    for page_id, owner_id, component_id in (
                        ("page_001", "owner_a", "component_a"),
                        ("page_003", "owner_c", "component_c"),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _selected_final_paths(
        {
            "targets": [
                {"page_id": "page_001", "owner_id": "owner_a", "component_ids": ["component_a"], "category": "burst"},
                {"page_id": "page_003", "owner_id": "owner_c", "component_ids": ["component_c"], "category": "burst"},
            ]
        },
        "burst",
        tmp_path,
    ) == [files[0], files[2]]


def test_contact_sheets_segment_tall_pages_at_readable_scale(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import _write_contact_sheets

    output_root = tmp_path / "matrix"
    work_dir = output_root / "entry_a"
    for folder in ("originals", "images", "translated"):
        (work_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 1000), "white").save(work_dir / folder / "001.png")
    (work_dir / "project.json").write_text(
        json.dumps(
            {
                "paginas": [{"numero": 1, "image_layers": {"rendered": {"path": "translated/001.png"}}}],
                "page_owner_graphs": [{"page_id": "page_001", "owners": [{"owner_id": "owner_a", "component_ids": ["component_a"]}]}],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "entry_id": "entry_a",
            "work_dir": "entry_a",
            "categories": ["white_balloon"],
            "targets": [{"page_id": "page_001", "owner_id": "owner_a", "component_ids": ["component_a"], "category": "white_balloon"}],
        }
    ]

    sheets = _write_contact_sheets(entries, output_root)

    assert len(sheets["white_balloon"]) >= 2
    with Image.open(sheets["white_balloon"][0]) as sheet:
        assert sheet.width == 600
        assert sheet.height > 100


def test_contact_sheets_include_every_declared_visual_category(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import _write_contact_sheets

    output_root = tmp_path / "matrix"
    work_dir = output_root / "entry_a"
    for folder in ("originals", "images", "translated"):
        (work_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(work_dir / folder / "001.png")
    (work_dir / "project.json").write_text(
        json.dumps(
            {
                "paginas": [{"numero": 1, "image_layers": {"rendered": {"path": "translated/001.png"}}}],
                "page_owner_graphs": [{"page_id": "page_001", "owners": [{"owner_id": "owner_a", "component_ids": ["component_a"]}]}],
            }
        ),
        encoding="utf-8",
    )
    entries = [
        {
            "entry_id": "entry_a",
            "work_dir": "entry_a",
            "categories": ["white_balloon", "sfx", "hard_negative"],
            "targets": [
                {"page_id": "page_001", "owner_id": "owner_a", "component_ids": ["component_a"], "category": category}
                for category in ("white_balloon", "sfx", "hard_negative")
            ],
        }
    ]

    sheets = _write_contact_sheets(entries, output_root)

    assert set(sheets) == {"hard_negative", "sfx", "white_balloon"}
    assert sheets["hard_negative"] == sheets["sfx"] == sheets["white_balloon"]


def test_contact_sheet_transforms_logical_owner_crop_into_framed_space(tmp_path):
    from PIL import Image
    from strip.page_surface_geometry import PageSurfaceGeometry
    from tools.validate_owner_visual_matrix import _write_contact_sheets

    output_root = tmp_path / "matrix"
    work_dir = output_root / "entry_narrow"
    for folder in ("originals", "images", "translated"):
        (work_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 160), "white").save(work_dir / folder / "001.png")
    geometry = PageSurfaceGeometry.build(
        logical_width=69,
        logical_height=160,
        frame_width=80,
        frame_height=160,
        content_origin_xy=(5, 0),
    )
    (work_dir / "project.json").write_text(json.dumps({
        "paginas": [{
            "numero": 1,
            "page_id": "page_001",
            "page_surface_geometry": geometry.to_dict(),
            "page_surface_geometry_sha256": geometry.geometry_sha256,
            "image_layers": {"rendered": {"path": "translated/001.png"}},
            "text_layers": [{
                "owner_id": "owner_a",
                "translated": "CORPO COMPLETO",
                "render_layout_contract": {
                    "coordinate_space": "logical_page",
                    "render_bbox": [10, 98, 32, 111],
                },
            }],
        }],
        "page_owner_graphs": [{
            "page_id": "page_001",
            "owners": [{"owner_id": "owner_a", "component_ids": ["component_a"]}],
            "components": [{
                "component_id": "component_a",
                "bbox_page": [10, 98, 32, 111],
                "polygon_page": [[10, 98], [32, 98], [32, 111], [10, 111]],
            }],
        }],
    }), encoding="utf-8")
    entries = [{
        "entry_id": "entry_narrow",
        "work_dir": "entry_narrow",
        "categories": ["cross_tile_owner"],
        "targets": [{
            "page_id": "page_001",
            "owner_id": "owner_a",
            "component_ids": ["component_a"],
            "category": "cross_tile_owner",
            "expected_artifact_space": "framed_page",
            "source_crop": {
                "bbox_page": [10, 98, 32, 111],
                "coordinate_space": "logical_page",
            },
        }],
    }]

    sheets = _write_contact_sheets(entries, output_root)
    metadata_path = Path(sheets["cross_tile_owner"][0]).with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["bbox_logical"] == [10, 98, 32, 111]
    assert metadata["artifact_bbox_frame"] == [15, 98, 37, 111]
    assert metadata["page_surface_geometry_sha256"] == geometry.geometry_sha256
    assert metadata["owner_map"]["polygons_frame"][0][0] == [15, 98]
    assert metadata["owner_map"]["polygons_frame"][0][2] == [37, 111]
    assert set(metadata["panels"]) == {
        "source", "masks_evidence", "requested_resolved",
        "observed_raster", "final", "safe_geometry",
    }
    assert all(panel["source_sha256"] for panel in metadata["panels"].values())
    assert metadata["panels"]["requested_resolved"]["plan_sha256"]
    assert metadata["panels"]["observed_raster"]["observation_sha256"]


def test_contact_sheet_blocks_logical_target_without_page_geometry(tmp_path):
    from PIL import Image
    from tools.validate_owner_visual_matrix import MatrixContractError, _write_contact_sheets

    output_root = tmp_path / "matrix"
    work_dir = output_root / "entry_missing"
    for folder in ("originals", "images", "translated"):
        (work_dir / folder).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 160), "white").save(work_dir / folder / "001.png")
    (work_dir / "project.json").write_text(json.dumps({
        "paginas": [{"numero": 1, "page_id": "page_001", "image_layers": {"rendered": {"path": "translated/001.png"}}}],
        "page_owner_graphs": [{"page_id": "page_001", "owners": [{"owner_id": "owner_a", "component_ids": ["component_a"]}]}],
    }), encoding="utf-8")
    entries = [{
        "entry_id": "entry_missing", "work_dir": "entry_missing",
        "categories": ["cross_tile_owner"],
        "targets": [{
            "page_id": "page_001", "owner_id": "owner_a",
            "component_ids": ["component_a"], "category": "cross_tile_owner",
            "expected_artifact_space": "framed_page",
            "source_crop": {"bbox_page": [10, 98, 32, 111], "coordinate_space": "logical_page"},
        }],
    }]

    with pytest.raises(MatrixContractError, match="missing_page_surface_geometry"):
        _write_contact_sheets(entries, output_root)


def test_inspection_template_deduplicates_identical_pixels_and_keeps_categories(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import build_inspection_template

    artifact = tmp_path / "same.png"
    Image.new("RGB", (12, 8), "white").save(artifact)

    template = build_inspection_template(
        {"speech": [str(artifact)], "white_balloon": [str(artifact)]},
        tmp_path,
    )

    assert len(template["artifacts"]) == 1
    assert template["artifacts"][0]["category"] == "speech"
    assert template["artifacts"][0]["categories"] == ["speech", "white_balloon"]


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


def test_runner_injects_runtime_models_dir_without_polluting_versioned_config(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.validate_owner_visual_matrix import _run_entry

    config = tmp_path / "fixture.json"
    config.write_text(json.dumps({"input_key": "fixture"}), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    captured = {}

    def fake_run(command, **_kwargs):
        if command[0] == "git":
            return SimpleNamespace(returncode=0, stdout="a" * 40 if "rev-parse" in command else "", stderr="")
        effective = Path(command[-1])
        captured.update(json.loads(effective.read_text(encoding="utf-8")))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tools.validate_owner_visual_matrix.subprocess.run", fake_run)
    _run_entry(
        {"entry_id": "entry_a", "work_dir": "entry_a"},
        tmp_path / "matrix.json",
        tmp_path / "output",
        {"config_path": config, "source_path": source},
    )

    assert Path(captured["models_dir"]).resolve() == (PIPELINE / "models").resolve()
    assert "models_dir" not in json.loads(config.read_text(encoding="utf-8"))


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


def test_style_matrix_requires_owner_paired_source_and_final(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result
    artifact = tmp_path / "final.png"; artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    layer = project["paginas"][0]["text_layers"][0]
    layer["visual_profile_v2"] = {"owner_id": "another", "source_sha256": "a" * 64}
    layer["style_v2_raster_contract"] = {"status": "applied", "applied_attributes": {"fill": "#fff"}}
    project["qa"]["style_fidelity"] = {"gate": {"status": "PASS"}, "owners": []}
    entry, _ = _write_owner_project(tmp_path, project)

    assert "style_owner_pair_mismatch" in validate_entry_result(entry, tmp_path)["contracts"]


def test_matrix_rejects_missing_owner_or_zero_owner_category(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    entry, work_dir = _write_owner_project(tmp_path, project)
    entry["categories"] = ["sfx"]
    entry["targets"] = [
        {
            "page_id": "page_001",
            "owner_id": "missing",
            "component_ids": ["component_missing"],
            "category": "sfx",
            "minimum_count": 1,
            "source_crop": {"bbox_page": [0, 0, 1, 1], "width": 1, "height": 1, "sha256": "a" * 64},
        }
    ]

    result = validate_entry_result(entry, tmp_path)

    assert result["status"] == "BLOCK"
    assert "owner_target_not_found" in result["contracts"]
    assert result["category_metrics"]["sfx"]["owner_count"] == 0


def test_style_matrix_reports_explicit_fallbacks(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result
    artifact = tmp_path / "final.png"; artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    layer = project["paginas"][0]["text_layers"][0]
    layer["visual_profile_v2"] = {"owner_id": "owner_a", "source_sha256": "a" * 64, "status": "fallback"}
    layer["style_v2_raster_contract"] = {"status": "fallback", "applied_attributes": {}}
    project["qa"]["style_fidelity"] = {"gate": {"status": "PASS"}, "owners": [{"owner_id": "owner_a"}]}
    entry, _ = _write_owner_project(tmp_path, project)

    result = validate_entry_result(entry, tmp_path)
    assert result["style_fallback_owner_ids"] == ["owner_a"]


def test_style_matrix_cannot_pass_when_functional_gate_is_blocked(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result
    artifact = tmp_path / "final.png"; artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    project["qa"]["export_gate"]["status"] = "BLOCK"
    project["qa"]["style_fidelity"] = {"gate": {"status": "PASS"}, "owners": []}
    entry, _ = _write_owner_project(tmp_path, project)

    assert validate_entry_result(entry, tmp_path)["status"] == "BLOCK"


def test_style_matrix_enforces_category_thresholds(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result
    artifact = tmp_path / "final.png"; artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    project["qa"]["style_fidelity"] = {
        "gate": {"status": "PASS"}, "owners": [],
        "category_metrics": {"colored_card": {"go_rate": 0.70}},
    }
    entry, _ = _write_owner_project(tmp_path, project)

    assert "style_category_below_threshold:colored_card" in validate_entry_result(entry, tmp_path)["contracts"]


def test_style_matrix_manifest_uses_validator_entries_schema():
    from tools.validate_owner_visual_matrix import validate_manifest
    path = PIPELINE / "tests" / "fixtures" / "style_copy_corpus" / "matrix.json"
    entries = validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    assert len(entries) == 3


def test_style_matrix_separates_calibration_and_holdout_works():
    path = PIPELINE / "tests" / "fixtures" / "style_copy_corpus" / "matrix.json"
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    calibration = {row["work_id"] for row in entries if row["split"] == "calibration"}
    holdout = {row["work_id"] for row in entries if row["split"] == "holdout"}
    assert calibration and holdout and calibration.isdisjoint(holdout)


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
        for target in entry["targets"]:
            target["split"] = "holdout"
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
    template = {"schema_version": 2, "artifacts": [{"artifact_path": "sheet.png", "sha256": sha256(b"sheet").hexdigest(), "category": "burst", "segment": "entry:1"}]}
    incomplete = {"schema_version": 2, "inspections": [{"artifact_path": "sheet.png"}]}
    with pytest.raises(MatrixContractError, match="inspection fields"):
        validate_inspection_manifest(template, incomplete, tmp_path)


def test_inspection_manifest_uses_plan_go_no_go_vocabulary(tmp_path):
    from tools.validate_owner_visual_matrix import validate_inspection_manifest

    artifact = tmp_path / "sheet.png"
    artifact.write_bytes(b"sheet")
    digest = sha256(b"sheet").hexdigest()
    template = {
        "schema_version": 2,
        "artifacts": [
            {
                "artifact_path": "sheet.png",
                "sha256": digest,
                "category": "burst",
                "segment": "entry:1",
            }
        ]
    }
    for verdict in ("GO", "NO-GO"):
        manifest = {
            "schema_version": 2,
            "inspections": [
                {
                    "artifact_path": "sheet.png",
                    "sha256": digest,
                    "scale": "native",
                    "timestamp": "2026-07-30T23:30:00-03:00",
                    "category": "burst",
                    "owner_or_segment": "entry:1",
                    "functional_verdict": verdict,
                    "style_verdict": verdict,
                    "functional_note": "funcional inspecionado em pixels nativos",
                    "style_note": "estilo inspecionado em pixels nativos",
                }
            ]
        }
        assert validate_inspection_manifest(template, manifest, tmp_path)[0][
            "overall_verdict"
        ] == verdict


def test_report_rejects_unverified_inspection_claims(tmp_path):
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_inspection_manifest

    artifact = tmp_path / "sheet.png"
    artifact.write_bytes(b"sheet")
    template = {"schema_version": 2, "artifacts": [{"artifact_path": "sheet.png", "sha256": sha256(b"sheet").hexdigest(), "category": "burst", "segment": "entry:1"}]}
    manifest = {"schema_version": 2, "inspections": [{"artifact_path": "sheet.png", "sha256": "0" * 64, "scale": "native", "timestamp": "2026-07-30T12:00:00Z", "category": "burst", "owner_or_segment": "entry:1", "functional_verdict": "GO", "style_verdict": "GO", "functional_note": "clean", "style_note": "clean"}]}
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
        inspected=[{"artifact_path": str(checked), "sha256": sha256(b"a").hexdigest(), "functional_verdict": "GO", "style_verdict": "GO", "overall_verdict": "GO", "functional_note": "native check", "style_note": "native check"}],
    )
    visual_section = report.read_text(encoding="utf-8").split("## Visually inspected", 1)[1]
    assert str(checked) in visual_section
    assert str(unchecked) not in visual_section


def test_incomplete_inspection_is_pending_not_go(tmp_path):
    from tools.validate_owner_visual_matrix import validate_inspection_manifest

    artifact = tmp_path / "sheet.png"
    artifact.write_bytes(b"sheet")
    template = {
        "schema_version": 2,
        "artifacts": [
            {
                "artifact_path": "sheet.png",
                "sha256": sha256(b"sheet").hexdigest(),
                "category": "burst",
                "segment": "owner_a",
            }
        ],
    }

    inspected = validate_inspection_manifest(
        template,
        {"schema_version": 2, "inspections": []},
        tmp_path,
    )

    assert inspected[0]["overall_verdict"] == "PENDING"


def test_runner_manifest_hash_tampering_is_rejected():
    from tools.validate_owner_visual_matrix import MatrixContractError, validate_runner_evidence

    evidence = {"schema_version": 1, "entry_id": "entry_a", "returncode": 0}
    canonical = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence["runner_evidence_sha256"] = sha256(canonical).hexdigest()
    validate_runner_evidence(evidence)
    evidence["returncode"] = 2

    with pytest.raises(MatrixContractError, match="runner manifest hash mismatch"):
        validate_runner_evidence(evidence)
