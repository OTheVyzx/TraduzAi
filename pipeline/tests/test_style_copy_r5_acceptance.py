from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from debug_tools.style_copy_r5_acceptance import (
    InitRunArgs, SOURCE_EXCLUSIONS, _sha256_input, evaluate_acceptance, init_run,
    load_active_manifest, main,
)


def _bundle() -> dict:
    return {
        "schema_version": 1,
        "acceptance_bundle_id": "b" * 64,
        "revision_sha256": "r" * 64,
        "source_manifest_sha256": "s" * 64,
        "matrix_sha256": "m" * 64,
    }


def _bound(run_id: str) -> dict:
    return {
        "producer_run_id": run_id,
        "acceptance_bundle_id": "b" * 64,
        "revision_sha256": "r" * 64,
        "source_manifest_sha256": "s" * 64,
    }


def _inputs() -> dict:
    synthetic = {
        "schema_version": 3, **_bound("benchmark-001"),
        "status": "PASS", "validation": {"status": "PASS"},
        "score": {
            "font_top1": {"evaluated": 20, "rate": 0.95},
            "font_top3": {"evaluated": 20, "rate": 1.0},
            "fill_delta_e_2000": {"count": 20, "median": 2.0, "p95": 4.0},
        },
    }
    owner = {
        "schema_version": 1, **_bound("owner-qa-001"), "status": "PASS",
        "eligible_owner_count": 9, "materialization_observation_coverage": 1.0,
        "safe_containment": {"evaluated": 9, "rate": 1.0},
        "effect_containment": {"evaluated": 9, "rate": 1.0},
        "catastrophic_mismatches": {"evaluated": 9, "count": 0},
    }
    inspection = {
        "schema_version": 1, **_bound("inspection-001"), "status": "PASS",
        "owners": {"evaluated": 5, "go_rate": 1.0},
        "speech": {"evaluated": 4, "go_rate": 1.0},
        "required_holdout_categories": ["white_balloon", "burst"],
        "categories": {
            "white_balloon": {"evaluated": 1, "go_rate": 1.0},
            "burst": {"evaluated": 1, "go_rate": 1.0},
        },
        "inspection_coverage": {"evaluated": 5, "rate": 1.0},
    }
    matrix = {
        "schema_version": 1, **_bound("matrix-001"),
        "functional_status": "GO", "style_status": "GO", "inspection_status": "GO",
        "entries": [{"entry_id": "entry-a", "export_gate_status": "PASS", "export_gate_sha256": "e" * 64}],
        "findings": [],
    }
    thresholds = {
        "schema_version": 2,
        "synthetic": {"font_top1_min": 0.9, "font_top3_min": 0.98, "fill_delta_e_2000_median_max": 8, "fill_delta_e_2000_p95_max": 12},
        "owner_qa": {"safe_containment_min": 1.0, "effect_containment_min": 1.0, "catastrophic_mismatch_max": 0},
        "inspection": {"owner_go_rate_min": 0.9, "speech_go_rate_min": 0.95, "category_go_rate_min": 0.85, "coverage_min": 1.0},
    }
    return {"acceptance_bundle": _bundle(), "synthetic": synthetic, "owner_qa": owner, "inspection": inspection, "matrix_execution": matrix, "thresholds": thresholds}


def test_acceptance_passes_only_with_complete_bound_evidence():
    result = evaluate_acceptance(**_inputs())

    assert result["status"] == "PASS"
    assert result["verdicts"] == {"style": "GO", "functional": "GO", "inspection": "GO", "overall": "GO"}


def test_distinct_producer_run_ids_pass_when_they_share_one_authenticated_bundle():
    inputs = _inputs()

    assert len({inputs[name]["producer_run_id"] for name in ("synthetic", "owner_qa", "inspection", "matrix_execution")}) == 4
    assert evaluate_acceptance(**inputs)["status"] == "PASS"


@pytest.mark.parametrize("problem", ["missing_metric", "zero_denominator", "bundle_mismatch", "threshold_breach", "pending_inspection", "functional_block"])
def test_acceptance_blocks_every_incomplete_or_divergent_input(problem):
    inputs = _inputs()
    if problem == "missing_metric":
        del inputs["synthetic"]["score"]["font_top1"]
    elif problem == "zero_denominator":
        inputs["owner_qa"]["safe_containment"]["evaluated"] = 0
    elif problem == "bundle_mismatch":
        inputs["inspection"]["acceptance_bundle_id"] = "x" * 64
    elif problem == "threshold_breach":
        inputs["synthetic"]["score"]["font_top1"]["rate"] = 0.2
    elif problem == "pending_inspection":
        inputs["inspection"]["status"] = "BLOCK"
    else:
        inputs["matrix_execution"]["functional_status"] = "BLOCK"

    result = evaluate_acceptance(**inputs)

    assert result["status"] == "BLOCK"
    assert result["verdicts"]["overall"] == "NO-GO"
    assert result["findings"]


def test_acceptance_cli_returns_two_on_block_and_writes_summary(tmp_path: Path):
    inputs = _inputs(); inputs["inspection"]["status"] = "BLOCK"
    paths = {}
    for name, payload in inputs.items():
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(payload), encoding="utf-8"); paths[name] = path
    output = tmp_path / "acceptance.json"

    exit_code = main([
        "evaluate", "--acceptance-bundle", str(paths["acceptance_bundle"]),
        "--benchmark-summary", str(paths["synthetic"]), "--owner-qa-summary", str(paths["owner_qa"]),
        "--inspection-summary", str(paths["inspection"]), "--matrix-execution-summary", str(paths["matrix_execution"]),
        "--thresholds", str(paths["thresholds"]), "--output", str(output), "--mode", "enforce",
    ])

    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "BLOCK"


def test_init_run_creates_fresh_root_bundle_and_active_manifest(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"; pipeline = repo / "pipeline"; pipeline.mkdir(parents=True)
    (pipeline / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_manifest = repo / "sources.json"
    source_manifest.write_text(json.dumps({
        "schema_version": 1, "closure_roots": ["pipeline"],
        "exclusions": list(SOURCE_EXCLUSIONS), "sources": ["pipeline/main.py"],
    }), encoding="utf-8")
    benchmark = repo / "benchmark.json"; benchmark.write_text("{}", encoding="utf-8")
    matrix = repo / "matrix.json"; matrix.write_text("{}", encoding="utf-8")
    source = tmp_path / "input"; source.mkdir(); (source / "page.png").write_bytes(b"page")
    monkeypatch.setenv("R5_INPUT", str(source))
    inputs = repo / "inputs.json"; inputs.write_text(json.dumps({
        "schema_version": 1, "hash_algorithm": "sha256-tree-v1",
        "inputs": {"fixture": {"environment_variable": "R5_INPUT", "input_type": "directory", "expected_sha256": _sha256_input(source)}},
    }), encoding="utf-8")
    active = tmp_path / "active.json"
    args = InitRunArgs(repo, tmp_path / "runs", "r5", source_manifest, benchmark, matrix, inputs, 1729, active)

    first = init_run(args); second = init_run(args)

    assert first.run_root != second.run_root
    assert first.acceptance_bundle_path.is_file()
    assert load_active_manifest(second.active_manifest_path)["acceptance_bundle_id"] == second.acceptance_bundle_id
