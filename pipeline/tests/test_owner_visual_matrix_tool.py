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
        "config_path": f"configs/{work_id}.json",
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
