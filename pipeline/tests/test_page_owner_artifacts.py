from __future__ import annotations

from dataclasses import replace
import json

import pytest

from main import _write_page_artifacts
from ownership.execution import PageArtifactIntegrityError, PageExecutionEvidenceSnapshot
from strip.page_pipeline import CanonicalVisualStage, PageExecutionResult
from test_chapter_publication import _generation, _verified_result


def _evidence(tmp_path):
    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    original_ref = result.request.original_page.artifact_ref
    final_ref = result.final_page.artifact_ref
    stages = (
        CanonicalVisualStage("original", original_ref, original_ref.pixel_sha256),
        CanonicalVisualStage(
            "inpaint", original_ref, original_ref.pixel_sha256,
            alias_of="original", alias_reason="fixture has no cleanup delta",
        ),
        CanonicalVisualStage("typeset", final_ref, final_ref.pixel_sha256),
        CanonicalVisualStage(
            "page_composition", final_ref, final_ref.pixel_sha256,
            alias_of="typeset", alias_reason="page-first composition is the typeset raster",
        ),
        CanonicalVisualStage(
            "persisted_final", final_ref, final_ref.pixel_sha256,
            alias_of="page_composition", alias_reason="lossless persistence preserves pixels",
        ),
    )
    result = PageExecutionResult.build_from(result, visual_stage_artifacts=stages)
    return PageExecutionEvidenceSnapshot.build(result), root, result


def test_page_artifacts_are_derived_from_reopened_snapshot_and_manifest_excludes_itself(tmp_path):
    evidence, root, result = _evidence(tmp_path)

    artifacts = _write_page_artifacts(evidence, generation_root=root)

    page_root = root / "evidence" / "pages" / "page_001"
    assert artifacts.execution.page_id == result.page_id
    assert (page_root / "page_execution_evidence.json").read_bytes() == evidence.canonical_json_bytes
    manifest = json.loads((page_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths = {item["relative_path"] for item in manifest["artifacts"]}
    assert "evidence/pages/page_001/artifact_manifest.json" not in paths
    assert "evidence/pages/page_001/page_execution_evidence.json" in paths
    assert artifacts.execution.result_sha256 == evidence.page_result_sha256


def test_page_artifacts_persist_all_canonical_visual_stages_as_verified_refs(tmp_path):
    evidence, root, _ = _evidence(tmp_path)

    artifacts = _write_page_artifacts(evidence, generation_root=root)

    assert set(artifacts.visual_stages) == {
        "original", "inpaint", "typeset", "page_composition", "persisted_final"
    }
    for stage in artifacts.visual_stages.values():
        loaded = stage.artifact_ref.load_verified(root)
        assert loaded.shape == (90, 140, 3)
        assert stage.pixel_sha256 == stage.artifact_ref.pixel_sha256
        if stage.alias_of is not None:
            assert stage.alias_reason
            assert stage.pixel_sha256 == artifacts.visual_stages[stage.alias_of].pixel_sha256


def test_page_artifact_writer_rejects_missing_stage_without_debug_reconstruction(tmp_path):
    evidence, root, result = _evidence(tmp_path)
    incomplete = PageExecutionResult.build_from(
        result, visual_stage_artifacts=()
    )
    broken = PageExecutionEvidenceSnapshot.build(incomplete)

    with pytest.raises(PageArtifactIntegrityError, match="visual stage"):
        _write_page_artifacts(broken, generation_root=root)

    assert not (root / "evidence" / "pages" / "page_001").exists()


def test_page_artifact_writer_rejects_tampered_snapshot_before_writing(tmp_path):
    evidence, root, _ = _evidence(tmp_path)
    tampered = replace(evidence, canonical_json_bytes=evidence.canonical_json_bytes + b" ")

    with pytest.raises(PageArtifactIntegrityError):
        _write_page_artifacts(tampered, generation_root=root)

    assert not (root / "evidence" / "pages" / "page_001").exists()
