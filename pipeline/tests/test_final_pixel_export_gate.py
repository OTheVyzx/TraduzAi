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
    "layout_legibility_contract": "PASS",
    "residual_cleanup_contract": "PASS",
    "protected_art_contract": "PASS",
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


def test_export_gate_blocks_detected_blocks_without_ocr(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    report = project["qa"]["final_pixel_reports"][0]
    report.update({
        "detected_block_count": 2,
        "ocr_record_count": 0,
        "coverage_complete": False,
        "coverage_failures": ["detector_blocks_without_usable_ocr"],
    })

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "detector_blocks_without_usable_ocr" in _reasons(gate)


def test_export_gate_blocks_missing_owner_render_quality_contract(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["paginas"][0]["text_layers"] = [{
        "id": "owner_a",
        "owner_id": "owner_a",
        "render_completed": True,
        "fit_status": "ok",
        "render_bbox": [10, 10, 30, 20],
    }]

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "missing_owner_render_quality_contract" in _reasons(gate)


def test_export_gate_blocks_under_source_scale_owner(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["paginas"][0]["text_layers"] = [{
        "id": "owner_a",
        "owner_id": "owner_a",
        "render_completed": True,
        "fit_status": "ok",
        "render_bbox": [10, 10, 30, 20],
        "owner_render_quality": {
            "schema_version": 1,
            "status": "under_source_scale",
            "source_scale_ratio": 0.60,
            "outside_safe_pixels": 0,
            "rendered_line_core_heights_px": [8],
        },
    }]

    gate = evaluate_export_gate(project)

    assert gate["status"] == "BLOCK"
    assert "under_source_scale" in _reasons(gate)


def _ok_quality():
    return {
        "schema_version": 1,
        "status": "ok",
        "source_scale_ratio": 1.0,
        "outside_safe_pixels": 0,
        "rendered_line_core_heights_px": [18],
    }


def test_export_gate_blocks_source_payload_incomplete(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    report = project["qa"]["final_pixel_reports"][0]
    report["contracts"]["source_coverage_contract"] = "BLOCK"
    report["issues"] = [{
        "issue_id": "source-incomplete",
        "page_id": "page_001",
        "owner_id": "owner_a",
        "component_ids": ["component_a"],
        "severity": "critical",
        "reason": "source_payload_incomplete",
        "offenders": ["200 MILLION GOLD"],
        "contract": "source_coverage_contract",
    }]

    gate = evaluate_export_gate(project)

    assert "source_payload_incomplete" in _reasons(gate)


def test_export_gate_blocks_unverified_owner_residual(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["paginas"][0]["text_layers"] = [{
        "owner_id": "owner_a",
        "component_ids": ["component_a"],
        "route_action": "translate_inpaint_render",
        "render_completed": True,
        "fit_status": "ok",
        "owner_render_quality": _ok_quality(),
        "residual_cleanup_contract": {"residual_verified": False},
        "protected_art_contract": {
            "protected_art_changed_pixels": 0,
            "action_protected_overlap_pixels": 0,
        },
    }]

    gate = evaluate_export_gate(project)

    assert "unverified_owner_residual" in _reasons(gate)


def test_export_gate_blocks_protected_art_contract_violation(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    project["paginas"][0]["text_layers"] = [{
        "owner_id": "owner_a",
        "component_ids": ["component_a"],
        "route_action": "translate_inpaint_render",
        "render_completed": True,
        "fit_status": "ok",
        "owner_render_quality": _ok_quality(),
        "residual_cleanup_contract": {
            "residual_verified": True,
            "residual_score": 0.0,
            "residual_threshold": 0.01,
        },
        "protected_art_contract": {
            "protected_art_changed_pixels": 4,
            "action_protected_overlap_pixels": 0,
        },
    }]

    gate = evaluate_export_gate(project)

    assert "protected_art_contract_violation" in _reasons(gate)


def test_export_gate_blocks_core_pixels_outside_safe_polygon(tmp_path):
    from qa.export_gate import evaluate_export_gate

    artifact = tmp_path / "001.png"
    artifact.write_bytes(b"page-one")
    project = _project(artifact)
    quality = _ok_quality()
    quality["outside_safe_pixels"] = 7
    project["paginas"][0]["text_layers"] = [{
        "owner_id": "owner_a",
        "render_completed": True,
        "fit_status": "ok",
        "owner_render_quality": quality,
    }]

    gate = evaluate_export_gate(project)

    assert "core_pixels_outside_safe_polygon" in _reasons(gate)
