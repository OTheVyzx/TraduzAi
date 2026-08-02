from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _graph() -> dict:
    return {
        "page_id": "page_001",
        "components": [{"component_id": "component_a", "bbox_page": [4, 5, 40, 28]}],
        "observations": [
            {
                "observation_id": "observation_a",
                "component_ids": ["component_a"],
                "text": "SOURCE BODY",
                "bbox_page": [4, 5, 40, 28],
            }
        ],
        "owners": [
            {
                "owner_id": "owner_a",
                "component_ids": ["component_a"],
                "observation_ids": ["observation_a"],
                "selected_observation_ids": ["observation_a"],
                "state": "rendered",
                "route_action": "translate_inpaint_render",
                "source_payload": "SOURCE BODY",
                "translated_payload": "CORPO TRADUZIDO",
            },
            {
                "owner_id": "owner_review",
                "component_ids": [],
                "observation_ids": [],
                "state": "review_required",
                "route_action": "review_required",
            },
        ],
        "component_dispositions": [
            {"component_id": "component_a", "decision": "owned", "owner_id": "owner_a"}
        ],
        "violations": [],
    }


def _publish(tmp_path: Path):
    from debug_tools import DebugRecorder
    from ownership.artifacts import OwnerArtifactPublisher

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-owner")
    publisher = OwnerArtifactPublisher(recorder)
    publisher.publish(
        graphs={"page_001": _graph()},
        executions=[
            {
                "page_id": "page_001",
                "owner_id": "owner_a",
                "coordinate_space": "page",
                "action_mask": np.full((32, 48), 255, dtype=np.uint8),
                "action_mask_sha256": "a" * 64,
                "before_sha256": "b" * 64,
                "after_sha256": "c" * 64,
            }
        ],
        compositions={
            "page_001": {
                "page_id": "page_001",
                "coordinate_space": "page",
                "sha256": "d" * 64,
                "write_counts": {"final_changed_pixels": 42},
                "owner_ids": ["owner_a"],
                "conflicts": [],
            }
        },
        final_pixel_reports=[
            {
                "page_id": "page_001",
                "persisted_sha256": "e" * 64,
                "ocr_records": [
                    {
                        "text": "CORPO TRADUZIDO",
                        "bbox": [4, 5, 40, 28],
                        "owner_id": "owner_a",
                    }
                ],
                "contracts": {"final_language_contract": "PASS"},
                "issues": [],
            }
        ],
        export_gate={"status": "PASS", "critical_issue_count": 0, "blocking_issue_count": 0, "issues": []},
    )
    return tmp_path / "debug" / "e2e"


def test_every_artifact_row_has_schema_run_page_owner_and_coordinate_space(tmp_path):
    root = _publish(tmp_path)
    row_paths = [
        "02_strip_detect/page_owner_components.jsonl",
        "03_ocr/page_owner_observations.jsonl",
        "04_text_normalization_router/source_evidence_ledger.jsonl",
        "09_typeset/owner_render_plan.jsonl",
        "10_copyback_reassemble/owner_composition.jsonl",
        "11_qa_export_gate/final_pixel_ocr.jsonl",
        "11_qa_export_gate/owner_pixel_checks.jsonl",
    ]
    for rel_path in row_paths:
        rows = _jsonl(root / rel_path)
        assert rows, rel_path
        for row in rows:
            assert row["schema_version"] == 1
            assert row["run_id"] == "run-owner"
            assert row["page_id"] == "page_001"
            assert "owner_id" in row
            assert row["trace_id"]
            assert row["coordinate_space"] == "page"
            assert isinstance(row["hashes"], dict) and row["hashes"]
            assert isinstance(row["offenders"], list)

    assert (root / "04_text_normalization_router/page_owner_graph.json").is_file()
    assert (root / "06_mask_segmentation/owner_masks/owner_a/action_mask.png").is_file()
    assert (root / "11_qa_export_gate/owner_invariant_report.json").is_file()


def test_rolled_back_execution_publishes_inpaint_evidence_for_diagnosis(tmp_path):
    from debug_tools import DebugRecorder
    from ownership.artifacts import OwnerArtifactPublisher

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-rollback")
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:9, 4:12] = 255
    before = np.full((12, 16, 3), 240, dtype=np.uint8)
    candidate = before.copy()
    candidate[mask > 0] = 200
    OwnerArtifactPublisher(recorder).publish(
        executions=[
            {
                "page_id": "page_002",
                "owner_id": "owner_rollback",
                "committed": False,
                "mutation": {
                    "action_mask": mask,
                    "changed_mask": mask,
                    "protected_art_mask": np.zeros_like(mask),
                    "result_rgb": candidate,
                },
            }
        ]
    )

    root = tmp_path / "debug" / "e2e" / "06_mask_segmentation" / "owner_masks" / "owner_rollback"
    assert (root / "action_mask.png").is_file()
    assert (root / "changed_mask.png").is_file()
    assert (root / "protected_art_mask.png").is_file()
    assert (root / "candidate_after_inpaint.png").is_file()


def test_source_evidence_ledger_is_derived_and_hash_linked(tmp_path):
    rows = _jsonl(
        _publish(tmp_path)
        / "04_text_normalization_router/source_evidence_ledger.jsonl"
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["observation_id"] == "observation_a"
    assert row["owner_id"] == "owner_a"
    assert row["disposition"] == "selected"
    assert row["material"] is True
    assert set(row["hashes"]) == {
        "graph_sha256",
        "observation_sha256",
    }


def test_render_plan_never_contains_review_required_owner(tmp_path):
    rows = _jsonl(_publish(tmp_path) / "09_typeset/owner_render_plan.jsonl")
    assert {row["owner_id"] for row in rows} == {"owner_a"}
    assert all(row["route_action"] != "review_required" for row in rows)


def test_all_qa_rows_have_composite_trace_and_offenders(tmp_path):
    from ownership.artifacts import validate_qa_rows

    rows = [
        {
            "page_id": "page_001",
            "owner_id": "owner_a",
            "trace_id": "page_001:owner_a:issue_a",
            "offenders": ["SOURCE BODY"],
            "severity": "critical",
        }
    ]
    assert validate_qa_rows(rows) == []
    assert "qa_row_trace_missing" in validate_qa_rows([{**rows[0], "trace_id": ""}])
    assert "qa_row_offenders_missing" in validate_qa_rows([{**rows[0], "offenders": []}])


def test_summary_row_and_gate_counts_match_exactly():
    from ownership.artifacts import validate_gate_integrity

    rows = [
        {
            "trace_id": "page_001:owner_a:issue_a",
            "offenders": ["SOURCE BODY"],
            "severity": "critical",
            "blocks_export": True,
        }
    ]
    summary = {"critical_issue_count": 1, "blocking_issue_count": 1}
    gate = {**summary, "issues": rows}
    assert validate_gate_integrity(summary=summary, gate=gate, rows=rows) == []
    assert "critical_issue_count_mismatch" in validate_gate_integrity(
        summary={**summary, "critical_issue_count": 0}, gate=gate, rows=rows
    )


def test_gate_integrity_counts_all_issues_but_scopes_owner_row_shape_contract():
    from ownership.artifacts import validate_gate_integrity

    owner_row = {
        "trace_id": "page_001:owner_a:issue_a",
        "offenders": ["SOURCE BODY"],
        "severity": "critical",
        "blocks_export": True,
    }
    legacy_row = {
        "trace_id": "",
        "offenders": [],
        "severity": "critical",
        "blocks_export": True,
    }
    rows = [owner_row, legacy_row]
    summary = {"critical_issue_count": 2, "blocking_issue_count": 2}
    gate = {**summary, "issues": rows}

    assert validate_gate_integrity(
        summary=summary,
        gate=gate,
        rows=rows,
        row_contract_rows=[owner_row],
    ) == []


def test_debug_artifacts_are_not_read_by_production_composer():
    source_path = Path(__file__).resolve().parents[1] / "compositor" / "owner_compositor.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not ({"open", "read_json", "read_jsonl"} & forbidden_calls)
    assert "ownership.artifacts" not in source_path.read_text(encoding="utf-8")
