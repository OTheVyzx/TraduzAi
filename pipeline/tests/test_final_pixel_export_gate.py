from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CONTRACTS = {
    "source_coverage_contract": "PASS",
    "owner_graph_contract": "PASS",
    "route_state_contract": "PASS",
    "pixel_ownership_contract": "PASS",
    "final_language_contract": "PASS",
    "qa_integrity_contract": "PASS",
}


def _project(path: Path, *, reports=True):
    report = {
        "page_id": "page_001",
        "artifact_path": str(path),
        "persisted_sha256": sha256(path.read_bytes()).hexdigest(),
        "observer": "DetectorOcrFinalPixelObserver",
        "observer_available": True,
        "observation_complete": True,
        "contracts": dict(CONTRACTS),
        "issues": [],
    }
    return {
        "owner_graph_status": "verified",
        "paginas": [{"numero": 1, "page_id": "page_001", "text_layers": []}],
        "qa": {"final_pixel_reports": [report] if reports else []},
    }


def _reasons(gate):
    return {
        issue.get("reason") or flag
        for issue in gate["issues"]
        for flag in (issue.get("flags") or [None])
    }


def test_automatic_export_requires_final_pixel_report_for_every_page(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    gate = evaluate_export_gate(_project(artifact, reports=False))

    assert gate["status"] == "BLOCK"
    assert "final_pixel_report_missing" in _reasons(gate)


def test_missing_or_unavailable_observer_blocks_export(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["qa"]["final_pixel_reports"][0]["observer_available"] = False
    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "final_pixel_observer_unavailable" in _reasons(gate)


def test_missing_or_stale_artifact_hash_blocks_export(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["qa"]["final_pixel_reports"][0]["persisted_sha256"] = "0" * 64
    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "final_pixel_artifact_hash_stale" in _reasons(gate)


def test_post_qa_image_mutation_invalidates_report(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    artifact.write_bytes(b"mutated-after-observer")
    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "final_pixel_artifact_hash_stale" in _reasons(gate)


def test_project_writer_rejects_mutation_after_passing_gate(tmp_path, monkeypatch):
    import project_writer

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["qa"]["export_gate"] = {"status": "PASS"}
    monkeypatch.setattr(project_writer, "require_owner_project_consistency", lambda _project: None)

    artifact.write_bytes(b"mutated-after-gate")

    with pytest.raises(ValueError, match="mudou apos o gate"):
        project_writer.validate_project_consistency(project)


def test_export_gate_identity_uses_owner_id_without_band_fallback(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    report = project["qa"]["final_pixel_reports"][0]
    report["contracts"]["final_language_contract"] = "BLOCK"
    report["issues"] = [
        {
            "issue_id": "page_001:source-visible",
            "page_id": "page_001",
            "owner_id": "owner_a",
            "component_ids": ["component_a"],
            "severity": "critical",
            "reason": "source_payload_visible",
            "offenders": ["SOURCE BODY"],
            "contract": "final_language_contract",
        }
    ]

    gate = evaluate_export_gate(project)
    issue = next(item for item in gate["issues"] if item.get("owner_id") == "owner_a")

    assert issue["page_id"] == "page_001"
    assert "band_id" not in issue


def test_summary_rows_offenders_and_gate_counts_are_identical(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    report = project["qa"]["final_pixel_reports"][0]
    report["contracts"]["final_language_contract"] = "BLOCK"
    report["issues"] = [
        {
            "issue_id": "issue_a",
            "page_id": "page_001",
            "owner_id": "owner_a",
            "component_ids": ["component_a"],
            "severity": "critical",
            "reason": "source_payload_visible",
            "offenders": ["SOURCE BODY"],
            "contract": "final_language_contract",
        }
    ]

    gate = evaluate_export_gate(project)
    final_rows = [item for item in gate["issues"] if item.get("source") == "final_pixel_qa"]

    assert len(final_rows) == len(report["issues"])
    assert final_rows[0]["offenders"] == report["issues"][0]["offenders"]
    assert gate["critical_issue_count"] == len(final_rows)
    assert gate["blocking_issue_count"] == len(final_rows)


def test_render_failure_propagates_instead_of_counting_success(tmp_path, monkeypatch):
    import main
    import typesetter.renderer

    project = {
        "_work_dir": str(tmp_path),
        "paginas": [
            {
                "numero": 1,
                "arquivo_original": "001.png",
                "text_layers": [{"translated": "TEXTO", "bbox": [1, 1, 10, 10]}],
            }
        ],
    }
    base = tmp_path / "base.png"
    base.write_bytes(b"base")
    monkeypatch.setattr(main, "_prepare_inpaint_base_for_render", lambda **_kwargs: base)
    monkeypatch.setattr(
        typesetter.renderer,
        "_typeset_single_page",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    try:
        main.render_page_image(project, 0, str(tmp_path / "out.png"))
    except RuntimeError as exc:
        assert "render failed" in str(exc)
    else:
        raise AssertionError("render failure was swallowed")
