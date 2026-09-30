from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

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


def test_matrix_cli_exposes_explicit_producer_run_id():
    result = subprocess.run(
        [sys.executable, str(PIPELINE / "tools" / "validate_owner_visual_matrix.py"), "--help"],
        cwd=PIPELINE.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-id RUN_ID" in result.stdout


def test_matrix_cli_exposes_external_auditor_contract():
    result = subprocess.run(
        [sys.executable, str(PIPELINE / "tools" / "validate_owner_visual_matrix.py"), "--help"],
        cwd=PIPELINE.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--audit-dir AUDIT_DIR" in result.stdout
    assert "--require-external-audit" in result.stdout


def test_validate_owner_visual_matrix_cli_forwards_external_audit_flags_and_exit_code(
    tmp_path, monkeypatch
):
    from tools import validate_owner_visual_matrix as matrix

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="PASS")

    monkeypatch.setattr(matrix, "validate_matrix_cli_request", fake, raising=False)
    paths = {
        "manifest": tmp_path / "functional_matrix.json",
        "output_root": tmp_path / "matrix-out",
        "report": tmp_path / "matrix.md",
        "inspection_template": tmp_path / "inspection.json",
        "audit_dir": tmp_path / "audits",
    }
    assert matrix.main([
        "--manifest", str(paths["manifest"]), "--output-root", str(paths["output_root"]),
        "--report", str(paths["report"]), "--inspection-template", str(paths["inspection_template"]),
        "--audit-dir", str(paths["audit_dir"]), "--require-external-audit",
        "--run-id", "universal-source-replacement-v1",
    ]) == 0
    assert calls == [{
        **{key: value.resolve() for key, value in paths.items()},
        "validate_only": False,
        "inspection_manifest": None,
        "acceptance_bundle": None,
        "require_external_audit": True,
        "run_id": "universal-source-replacement-v1",
    }]


def test_validate_owner_visual_matrix_cli_returns_nonzero_when_validation_raises(
    tmp_path, monkeypatch
):
    from tools import validate_owner_visual_matrix as matrix

    monkeypatch.setattr(
        matrix,
        "validate_matrix_cli_request",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("external audit missing")),
        raising=False,
    )
    assert matrix.main([
        "--manifest", str(tmp_path / "functional_matrix.json"),
        "--output-root", str(tmp_path / "matrix-out"),
        "--report", str(tmp_path / "matrix.md"),
        "--inspection-template", str(tmp_path / "inspection.json"),
        "--audit-dir", str(tmp_path / "audits"), "--require-external-audit",
        "--run-id", "universal-source-replacement-v1",
    ]) != 0


def test_visual_matrix_launches_auditor_cli_in_distinct_real_process_per_entry(tmp_path):
    from tools.validate_owner_visual_matrix import _run_external_audits

    script = tmp_path / "fake_auditor.py"
    log = tmp_path / "processes.jsonl"
    script.write_text(
        "import argparse, json, os\n"
        "p=argparse.ArgumentParser(); p.add_argument('--source'); p.add_argument('--run'); "
        "p.add_argument('--report'); p.add_argument('--review-dir'); p.add_argument('--source-lang'); "
        "p.add_argument('--target-lang'); p.add_argument('--require-final-verified', action='store_true'); a=p.parse_args()\n"
        "entry=os.environ['TRADUZAI_MATRIX_ENTRY_ID']; pid=os.getpid()\n"
        f"open({str(log)!r}, 'a', encoding='utf-8').write(json.dumps({{'entry_id':entry,'pid':pid}})+'\\n')\n"
        "audit={'page_id':'page_001','status':'PASS','auditor_invocation_id':'inv:'+entry,"
        "'auditor_execution_id':'exec:'+entry,'auditor_process_nonce':'nonce:'+entry}\n"
        "payload={'schema_version':1,'external_audit_gate_status':'PASS',"
        "'source_manifest_page_ids':['page_001'],'external_page_audits':[audit],"
        "'external_page_audits_count':1}\n"
        "open(a.report,'w',encoding='utf-8').write(json.dumps(payload))\n",
        encoding="utf-8",
    )
    entries, runtimes = [], {}
    for ordinal in (1, 2):
        entry_id = f"entry_{ordinal}"
        source, work = tmp_path / f"source_{ordinal}", tmp_path / f"work_{ordinal}"
        source.mkdir(); work.mkdir()
        config = tmp_path / f"config_{ordinal}.json"
        config.write_text(json.dumps({"idioma_origem": "en", "idioma_destino": "pt-BR"}), encoding="utf-8")
        entries.append({"entry_id": entry_id, "work_dir": str(work)})
        runtimes[entry_id] = {"source_path": source, "config_path": config}

    records = _run_external_audits(
        entries, runtimes, tmp_path / "out", tmp_path / "audits",
        external_auditor_command=(sys.executable, str(script)),
    )

    invocations = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert set(records) == {"entry_1", "entry_2"}
    assert {item["entry_id"] for item in invocations} == set(records)
    assert len({item["pid"] for item in invocations}) == 2


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


def test_resolved_intent_uses_render_layout_contract_not_source_owner_geometry():
    from strip.page_surface_geometry import PageSurfaceGeometry
    from tools.validate_owner_visual_matrix import _resolved_intent_panel

    geometry = PageSurfaceGeometry.build(
        logical_width=69,
        logical_height=160,
        frame_width=80,
        frame_height=160,
        content_origin_xy=(5, 0),
    )
    page = {
        "text_layers": [{
            "owner_id": "owner_a",
            "translated": "CORPO COMPLETO",
            "coordinate_space": "page",
            "owner_render_geometry": {
                "logical_space": "logical_page",
                "semantic_body_bbox_page": [8, 90, 34, 116],
            },
            "render_layout_contract": {
                "coordinate_space": "logical_page",
                "safe_text_box": [10, 98, 32, 111],
            },
        }],
    }

    _, metadata = _resolved_intent_panel(
        page,
        {"owner_id": "owner_a"},
        (80, 160),
        geometry,
    )

    assert metadata["render_bbox_frame"] == [15, 98, 37, 111]


def test_resolved_intent_audits_review_owner_from_logical_render_geometry():
    from strip.page_surface_geometry import PageSurfaceGeometry
    from tools.validate_owner_visual_matrix import _resolved_intent_panel

    geometry = PageSurfaceGeometry.build(
        logical_width=69,
        logical_height=160,
        frame_width=80,
        frame_height=160,
        content_origin_xy=(5, 0),
    )
    page = {
        "text_layers": [{
            "owner_id": "owner_review",
            "translated": "CORPO EM REVISAO",
            "route_action": "review_required",
            "owner_render_geometry": {
                "logical_space": "logical_page",
                "layout_container_bbox_page": [10, 98, 32, 111],
                "geometry_sha256": "a" * 64,
                "status": "ready",
            },
        }],
    }

    _, metadata = _resolved_intent_panel(
        page,
        {"owner_id": "owner_review"},
        (80, 160),
        geometry,
    )

    assert metadata["render_bbox_frame"] == [15, 98, 37, 111]
    assert metadata["plan_sha256"] == "a" * 64


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


def test_inspection_template_rejects_distinct_outputs_with_identical_pixels(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import MatrixContractError, build_inspection_template

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (12, 8), "white").save(first)
    second.write_bytes(first.read_bytes())

    with pytest.raises(MatrixContractError, match="distinct output artifacts"):
        build_inspection_template(
            {"speech": [str(first)], "white_balloon": [str(second)]}, tmp_path
        )


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


def test_runner_evidence_is_persisted_outside_published_work_dir(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.validate_owner_visual_matrix import _run_entry

    config = tmp_path / "fixture.json"
    config.write_text(json.dumps({"input_key": "fixture"}), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    output_root = tmp_path / "output"

    def fake_run(command, **_kwargs):
        if command[0] == "git":
            return SimpleNamespace(
                returncode=0,
                stdout="a" * 40 if "rev-parse" in command else "",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tools.validate_owner_visual_matrix.subprocess.run", fake_run)
    result = _run_entry(
        {"entry_id": "entry_a", "work_dir": "entry_a"},
        tmp_path / "matrix.json",
        output_root,
        {"config_path": config, "source_path": source},
    )

    runner_manifest_path = Path(result["runner_manifest_path"])
    assert runner_manifest_path == (
        output_root / "runner_evidence" / "entry_a.json"
    ).resolve()
    assert runner_manifest_path.is_file()
    assert not (output_root / "entry_a" / "run_manifest.json").exists()


def test_runner_evidence_hashes_canonical_published_final_artifacts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.validate_owner_visual_matrix import build_runner_evidence

    work_dir = tmp_path / "published"
    final = work_dir / ".page-generations" / "page_001" / "generation-1" / "final.png"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"verified-final")
    (work_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_id": "page_001",
                        "translated_path": ".page-generations/page_001/generation-1/final.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "project.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "tools.validate_owner_visual_matrix.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="a" * 40, stderr=""),
    )
    evidence = build_runner_evidence(
        entry={"entry_id": "entry_a", "work_dir": str(work_dir)},
        command=["python", "main.py"],
        completed=SimpleNamespace(returncode=0, stdout="", stderr=""),
        runtime={"input_sha256": "b" * 64, "config_path": str(config)},
        effective_config_path=config,
        started_at="2026-08-10T00:00:00Z",
        finished_at="2026-08-10T00:01:00Z",
    )

    assert evidence["final_artifact_sha256"] == {
        ".page-generations/page_001/generation-1/final.png": sha256(
            b"verified-final"
        ).hexdigest()
    }


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


def test_matrix_respects_ok_quality_when_source_scale_is_untrusted(tmp_path):
    from tools.validate_owner_visual_matrix import validate_entry_result

    artifact = tmp_path / "final.png"
    artifact.write_bytes(b"pixels")
    project = _passing_owner_project(artifact)
    quality = project["paginas"][0]["text_layers"][0]["owner_render_quality"]
    quality.update({"status": "ok", "source_scale_ratio": 0.26})
    entry, _ = _write_owner_project(tmp_path, project)

    result = validate_entry_result(entry, tmp_path)

    assert "under_source_scale" not in result["contracts"]
    assert result["status"] == "PASS"


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


def test_inspection_manifest_v3_rejects_template_notes_without_real_review_metadata(tmp_path):
    from PIL import Image

    from tools.validate_owner_visual_matrix import (
        InspectionEvidenceError,
        validate_inspection_manifest,
    )

    source = tmp_path / "source.png"
    output = tmp_path / "sheet.png"
    Image.new("RGB", (12, 8), "white").save(source)
    Image.new("RGB", (12, 8), "black").save(output)
    source_digest = sha256(source.read_bytes()).hexdigest()
    output_digest = sha256(output.read_bytes()).hexdigest()
    template = {
        "schema_version": 3,
        "artifacts": [{
            "artifact_path": "sheet.png", "sha256": output_digest,
            "category": "burst", "categories": ["burst"], "segment": "entry:1",
            "width": 12, "height": 8,
        }],
    }
    manifest = {
        "schema_version": 3,
        "inspections": [{
            "artifact_path": "sheet.png", "sha256": output_digest,
            "scale": "native", "timestamp": "2026-08-05T12:00:00-03:00",
            "reviewed_at": "2026-08-05T12:00:00-03:00", "reviewer": "codex-visual",
            "category": "burst", "owner_or_segment": "entry:1",
            "source_path": "source.png", "output_path": "sheet.png",
            "source_sha256": source_digest, "output_sha256": output_digest,
            "width": 12, "height": 8,
            "functional_verdict": "GO", "style_verdict": "GO",
            "functional_note": "clean", "style_note": "clean",
        }],
    }

    with pytest.raises(InspectionEvidenceError, match="independent review"):
        validate_inspection_manifest(template, manifest, tmp_path)


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


def _summary_matrix() -> dict:
    required = ("white_balloon", "burst", "cross_tile_owner", "dark_panel", "text_over_art")
    entries = [{
        "entry_id": "calibration", "split": "calibration",
        "targets": [{
            "page_id": "page_001", "owner_id": "owner_cal", "category": "colored_card",
            "split": "calibration", "semantic_role": "card_body", "is_speech": False,
            "source_crop": {"sha256": "1" * 64},
        }],
    }]
    for index, category in enumerate(required):
        entries.append({
            "entry_id": f"holdout_{index}", "split": "holdout",
            "targets": [{
                "page_id": "page_001", "owner_id": f"owner_{index}", "category": category,
                "split": "holdout", "semantic_role": "dialogue_body" if index < 2 else "card_body",
                "is_speech": index < 2, "source_crop": {"sha256": f"{index + 2:x}" * 64},
            }],
        })
    return {"schema_version": 3, "entries": entries}


def _complete_inspection_rows(matrix: dict) -> list[dict]:
    rows = []
    for entry in matrix["entries"]:
        for target in entry["targets"]:
            rows.append({
                "entry_id": entry["entry_id"], "page_id": target["page_id"],
                "owner_id": target["owner_id"], "category": target["category"],
                "split": target["split"], "source_sha256": target["source_crop"]["sha256"],
                "final_sha256": "a" * 64, "page_surface_geometry_sha256": "b" * 64,
                "artifact_path": f"{entry['entry_id']}.png", "overall_verdict": "GO",
                "functional_verdict": "GO", "style_verdict": "GO",
            })
    return rows


def test_inspection_summary_uses_exact_holdout_targets_not_filenames():
    from tools.validate_owner_visual_matrix import build_style_holdout_summary

    matrix = _summary_matrix()
    summary = build_style_holdout_summary(matrix=matrix, inspection=_complete_inspection_rows(matrix))

    assert summary["owners"]["evaluated"] == 5
    assert summary["calibration_owner_count"] == 1
    assert summary["owners"]["evaluated"] + summary["calibration_owner_count"] == 6


def test_inspection_summary_emits_owner_speech_and_category_go_rates():
    from tools.validate_owner_visual_matrix import build_style_holdout_summary

    matrix = _summary_matrix()
    summary = build_style_holdout_summary(matrix=matrix, inspection=_complete_inspection_rows(matrix))

    assert summary["owners"]["go_rate"] == 1.0
    assert summary["speech"]["evaluated"] == 2
    assert summary["speech"]["go_rate"] == 1.0
    assert set(summary["required_holdout_categories"]) == {
        "white_balloon", "burst", "cross_tile_owner", "dark_panel", "text_over_art",
    }
    assert all(summary["categories"][name]["evaluated"] > 0 for name in summary["required_holdout_categories"])


def _sealed_style_report(owner_id: str) -> dict:
    report = {
        "schema_version": 3, "run_id": f"run-{owner_id}",
        "summary": {"eligible_owner_count": 1, "rendered_owner_count": 1},
        "coverage": {"profile": 1.0, "contract": 1.0, "metrics": 1.0},
        "metrics": {
            "safe_containment": {"evaluated": 1, "contained": 1, "rate": 1.0},
            "effect_containment": {"evaluated": 1, "contained": 1, "rate": 1.0},
            "catastrophic_mismatches": {"evaluated": 1, "count": 0},
        },
        "owners": [{
            "page_id": "page_001", "owner_id": owner_id,
            "visual_profile_sha256": "c" * 64, "raster_contract_sha256": "d" * 64,
            "materialization_plan_sha256": "e" * 64,
            "materialization_observation_sha256": "f" * 64,
            "delivery_contract_sha256": "9" * 64,
            "owner_render_geometry_sha256": "8" * 64,
        }],
        "gate": {"status": "PASS"}, "findings": [],
    }
    report["report_sha256"] = sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report


def test_matrix_aggregates_bound_owner_qa_reports():
    from tools.validate_owner_visual_matrix import build_owner_qa_summary

    matrix = _summary_matrix()
    reports = {
        entry["entry_id"]: _sealed_style_report(entry["targets"][0]["owner_id"])
        for entry in matrix["entries"]
    }
    summary = build_owner_qa_summary(matrix=matrix, entry_reports=reports)

    assert summary["eligible_owner_count"] == 6
    assert summary["materialization_observation_coverage"] == 1.0
    assert summary["safe_containment"]["evaluated"] == 6
    assert summary["source_reports_sha256"]


@pytest.mark.parametrize("problem", ["pending", "missing", "duplicate", "hash_mismatch", "zero_category"])
def test_inspection_summary_blocks_incomplete_or_unauthenticated_denominator(problem):
    from tools.validate_owner_visual_matrix import build_style_holdout_summary

    matrix = _summary_matrix()
    rows = _complete_inspection_rows(matrix)
    if problem == "pending":
        rows[1]["overall_verdict"] = rows[1]["style_verdict"] = "PENDING"
    elif problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(dict(rows[-1]))
    elif problem == "hash_mismatch":
        rows[1]["source_sha256"] = "f" * 64
    else:
        matrix["entries"] = [entry for entry in matrix["entries"] if entry["targets"][0]["category"] != "burst"]
        rows = _complete_inspection_rows(matrix)

    summary = build_style_holdout_summary(matrix=matrix, inspection=rows)

    assert summary["status"] == "BLOCK"
    assert summary["findings"]
