from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ownership.execution import (
    ArtifactGenerationMarker,
    FinalPageSnapshot,
    FrozenJSONSnapshot,
    PageGeometrySnapshot,
    PageCandidateTransaction,
    PageArtifactIntegrityError,
    PageNotTerminalError,
    PersistedAssetRef,
    PersistedRGBImageArtifactRef,
    TerminalPixelProof,
)
from ownership.chapter_contract import ChapterSourceManifest, VerifiedProjectInputs
from ownership.publication import (
    PublicationBusyError,
    PublicationLock,
    PublicationRecoveryError,
    PublicationTransaction,
    recover_chapter_publication,
    reopen_verified_publication,
)
from strip.page_pipeline import (
    OriginalPageSnapshot,
    PageExecutionResult,
    adapt_page_execution_result_to_output_page,
    run_page_owner_pipeline,
)
from test_page_owner_pipeline import _page, _request, _services


def _generation(tmp_path: Path) -> tuple[Path, ArtifactGenerationMarker]:
    root = tmp_path / "generation"
    root.mkdir()
    marker = ArtifactGenerationMarker.create(
        artifact_store_id="store-a",
        generation_id="generation-a",
        run_id="run-a",
        execution_id="execution-a",
    )
    marker.write(root)
    return root, marker


def test_rgb_artifact_ref_loads_only_hash_linked_relative_asset(tmp_path):
    root, marker = _generation(tmp_path)
    path = root / "images" / "page_001.png"
    path.parent.mkdir(parents=True)
    pixels = np.full((12, 18, 3), (30, 60, 90), dtype=np.uint8)
    Image.fromarray(pixels, "RGB").save(path)
    ref = PersistedRGBImageArtifactRef.from_file(
        root,
        "images/page_001.png",
        marker=marker,
        page_id="page_001",
    )

    loaded = ref.load_verified(root)

    assert np.array_equal(loaded, pixels)
    assert loaded.flags.writeable is False


def test_rgb_artifact_ref_rejects_byte_or_pixel_tamper(tmp_path):
    root, marker = _generation(tmp_path)
    path = root / "images" / "page_001.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), (1, 2, 3)).save(path)
    ref = PersistedRGBImageArtifactRef.from_file(
        root, "images/page_001.png", marker=marker, page_id="page_001"
    )
    Image.new("RGB", (8, 6), (4, 5, 6)).save(path)

    with pytest.raises(PageArtifactIntegrityError, match="hash|pixel"):
        ref.load_verified(root)


@pytest.mark.parametrize(
    "bad_path",
    ["../escape.png", "/absolute.png", "C:/absolute.png", "images\\page.png", "images/./page.png"],
)
def test_artifact_ref_rejects_absolute_traversal_or_noncanonical_separator(tmp_path, bad_path):
    root, marker = _generation(tmp_path)
    payload = {
        "run_id": marker.run_id,
        "execution_id": marker.execution_id,
        "origin_execution_id": marker.execution_id,
        "source_artifact_ref_sha256": None,
        "page_id": "page_001",
        "artifact_store_id": marker.artifact_store_id,
        "generation_id": marker.generation_id,
        "relative_path": bad_path,
        "file_sha256": "a" * 64,
        "size_bytes": 1,
        "media_type": "application/octet-stream",
        "decoded_mode": None,
        "decoded_pixel_sha256": None,
        "width": None,
        "height": None,
    }

    with pytest.raises(PageArtifactIntegrityError):
        PersistedAssetRef.build(**payload)


def test_generation_marker_rejects_unknown_schema_or_rehashed_identity(tmp_path):
    root, marker = _generation(tmp_path)
    marker_path = root / "artifact_generation.json"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    marker_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PageArtifactIntegrityError, match="schema"):
        ArtifactGenerationMarker.read_verified(root)


def test_frozen_json_and_geometry_snapshots_return_fresh_copies():
    source = {"texts": [{"translated": "OLÁ"}]}
    snapshot = FrozenJSONSnapshot.build(source)
    geometry = PageGeometrySnapshot.build(
        run_id="run-a",
        execution_id="execution-a",
        page_id="page_001",
        page_source_sha256="a" * 64,
        y_top=0,
        y_bottom=100,
        width=80,
        height=100,
    )
    source["texts"][0]["translated"] = "MUTADO"
    first = snapshot.read()
    first["texts"][0]["translated"] = "OUTRO"

    assert snapshot.read()["texts"][0]["translated"] == "OLÁ"
    assert geometry.read()["width"] == 80


def test_final_page_snapshot_is_read_only_and_hash_linked_to_ref(tmp_path):
    root, marker = _generation(tmp_path)
    path = root / "translated" / "page_001.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (9, 7), (22, 44, 66)).save(path)
    ref = PersistedRGBImageArtifactRef.from_file(
        root, "translated/page_001.png", marker=marker, page_id="page_001"
    )
    snapshot = FinalPageSnapshot.from_artifact(ref, root)

    assert snapshot.page_output_pixel_sha256 == ref.pixel_sha256
    with pytest.raises(ValueError):
        snapshot.read_only_rgb()[0, 0] = 0


def _verified_result(root: Path, marker: ArtifactGenerationMarker) -> PageExecutionResult:
    original_path = root / "originals" / "page_001.png"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_page(), "RGB").save(original_path)
    original_ref = PersistedRGBImageArtifactRef.from_file(
        root, "originals/page_001.png", marker=marker, page_id="page_001"
    )
    original = OriginalPageSnapshot.from_pixels(
        _page(), source_file_sha256="f" * 64, artifact_ref=original_ref
    )
    base_request = _request()
    request = replace(
        base_request,
        run_id=marker.run_id,
        execution_id=marker.execution_id,
        original_page=original,
        page_source_sha256=original.page_source_sha256,
    )
    candidate = run_page_owner_pipeline(request, _services())
    final_path = root / "translated" / "page_001.png"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_page(), "RGB").save(final_path)
    final_ref = PersistedRGBImageArtifactRef.from_file(
        root, "translated/page_001.png", marker=marker, page_id="page_001"
    )
    final_page = FinalPageSnapshot.from_artifact(final_ref, root)
    bindings = candidate.translations
    proof = TerminalPixelProof.build(
        run_id=request.run_id,
        execution_id=request.execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        final_page_pixel_sha256=final_ref.pixel_sha256,
        cleanup_base_sha256="1" * 64,
        composition_sha256="2" * 64,
        translation_binding_sha256s=tuple(item.translation_binding_sha256 for item in bindings),
        source_payload_sha256s=tuple(item.source_payload_sha256 for item in bindings),
        target_payload_sha256s=tuple(item.target_payload_sha256 for item in bindings),
        target_glyph_patch_sha256s=(),
        target_materialization_sha256s=(),
        final_replacement_verdict_sha256s=(),
        repair_budget_policy_sha256="3" * 64,
        replacement_verification_policy_sha256="4" * 64,
        final_qa_probe_id="probe-final",
        fresh_ocr_invocation_id="invocation-final",
        fresh_ocr_root_input_pixel_sha256=final_ref.pixel_sha256,
        fresh_ocr_attempt_ids=("attempt-final",),
        fresh_ocr_attempt_chain_sha256="5" * 64,
        fresh_ocr_payload_sha256="6" * 64,
        source_support_removed=True,
        target_glyph_patch_applied=True,
        fresh_source_ocr_absent=True,
        coverage_complete=True,
        unowned_material_text_absent=True,
    )
    return PageExecutionResult.build(
        request=request,
        coverage=candidate.coverage,
        owner_graph=candidate.owner_graph,
        translation_attempts=candidate.translation_attempts,
        translations=bindings,
        status="final_verified",
        final_page=final_page,
        terminal_proof=proof,
    )


def test_page_candidate_transaction_persists_pointer_and_reopens_verified_result(tmp_path):
    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    transaction = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    )

    ref = transaction.commit_verified_generation(result)
    reopened = ref.read_verified(root)

    assert reopened.status == "final_verified"
    assert reopened.result_sha256 == result.result_sha256
    assert ref.page_generation_id == "page-generation-001"


def test_page_candidate_transaction_rejects_candidate_result(tmp_path):
    root, marker = _generation(tmp_path)
    candidate = run_page_owner_pipeline(_request(), _services())
    transaction = PageCandidateTransaction(
        private_execution_root=root,
        run_id=marker.run_id,
        execution_id=marker.execution_id,
        page_id="page_001",
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="candidate",
        transaction_id="candidate",
    )

    with pytest.raises(PageNotTerminalError):
        transaction.commit_verified_generation(candidate)


@pytest.mark.parametrize("tamper", ["pointer", "evidence", "final_asset"])
def test_page_evidence_ref_rejects_tampered_pointer_snapshot_or_asset(tmp_path, tamper):
    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    if tamper == "pointer":
        (root / Path(ref.current_pointer_relative_path)).write_text("{}", encoding="utf-8")
    elif tamper == "evidence":
        (root / Path(ref.page_execution_evidence_relative_path)).write_text("{}", encoding="utf-8")
    else:
        Image.new("RGB", (140, 90), (0, 0, 0)).save(root / "translated" / "page_001.png")

    with pytest.raises(PageArtifactIntegrityError):
        ref.read_verified(root)


def test_project_adapter_uses_persisted_pointer_and_ignores_mutable_output_views(tmp_path):
    import main

    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    manifest = ChapterSourceManifest.from_extracted_pages(
        (root / "originals" / "page_001.png",),
        root,
        run_id=marker.run_id,
        execution_id=marker.execution_id,
    )
    output = adapt_page_execution_result_to_output_page(result, evidence_ref=ref)
    output.owner_page_result = None
    output.image[:] = 0
    output.ocr_result = {"texts": [{"translated": "CORROMPIDO"}]}
    output.text_layers = {"texts": [{"translated": "CORROMPIDO"}]}

    verified = main._project_inputs_from_output_pages(
        manifest, [output], private_execution_root=root
    )

    assert isinstance(verified, VerifiedProjectInputs)
    assert verified.pages[0].final_artifact.pixel_sha256 == result.final_page.page_output_pixel_sha256
    assert verified.pages[0].page_result_sha256 == result.result_sha256


def test_verified_project_inputs_reopens_from_portable_generation_only(tmp_path):
    import main

    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    manifest = ChapterSourceManifest.from_extracted_pages(
        (root / "originals" / "page_001.png",), root,
        run_id=marker.run_id, execution_id=marker.execution_id,
    )
    output = adapt_page_execution_result_to_output_page(result, evidence_ref=ref)
    verified = main._project_inputs_from_output_pages(
        manifest, [output], private_execution_root=root
    )
    (root / "chapter_source_manifest.json").write_bytes(manifest.canonical_json_bytes)
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "verified_project_inputs.json").write_bytes(
        verified.canonical_json_bytes
    )

    reopened = VerifiedProjectInputs.read_verified(root)

    assert reopened.sha256 == verified.sha256
    assert reopened.pages[0].page_result_sha256 == result.result_sha256
    assert reopened.pages[0].final_artifact.pixel_sha256 == result.final_page.page_output_pixel_sha256


def test_wrap_up_builds_self_contained_bundle_that_reopens_after_publication(tmp_path):
    import main

    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    manifest = ChapterSourceManifest.from_extracted_pages(
        (root / "originals" / "page_001.png",), root,
        run_id=marker.run_id, execution_id=marker.execution_id,
    )
    inputs = main._project_inputs_from_output_pages(
        manifest,
        [adapt_page_execution_result_to_output_page(result, evidence_ref=ref)],
        private_execution_root=root,
    )

    bundle = main._wrap_up_verified_owner_pages(
        inputs, source_private_execution_root=root
    )
    staged = reopen_verified_publication(bundle.runtime_staging_root)
    public = tmp_path / "published"
    transaction = PublicationTransaction.for_bundle(public, bundle)
    transaction.stage(bundle)
    transaction.commit(bundle)
    reopened = reopen_verified_publication(public)

    assert staged.receipt.receipt_sha256 == bundle.publication_receipt.receipt_sha256
    assert reopened.verified_inputs.sha256 == bundle.verified_inputs_sha256
    assert reopened.export_manifest.pages[0].translated_path == "translated/page_001.png"


def test_verified_strip_boundary_calls_run_chapter_once_then_returns_only_frozen_inputs(tmp_path):
    import main

    root, marker = _generation(tmp_path)
    result = _verified_result(root, marker)
    ref = PageCandidateTransaction(
        private_execution_root=root,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-001",
        transaction_id="transaction-001",
    ).commit_verified_generation(result)
    manifest = ChapterSourceManifest.from_extracted_pages(
        (root / "originals" / "page_001.png",), root,
        run_id=marker.run_id, execution_id=marker.execution_id,
    )
    output = adapt_page_execution_result_to_output_page(result, evidence_ref=ref)
    calls = []

    def run_chapter_fn(**kwargs):
        calls.append(kwargs)
        return [output]

    verified = main._run_verified_strip_chapter(
        manifest,
        private_execution_root=root,
        run_chapter_fn=run_chapter_fn,
        image_files=[root / "originals" / "page_001.png"],
    )

    assert len(calls) == 1
    assert calls[0]["source_manifest"] is manifest
    assert isinstance(verified, VerifiedProjectInputs)
    assert not any(type(value).__name__ == "OutputPage" for value in verified.pages)


def test_second_live_writer_cannot_interleave_publication(tmp_path):
    root = tmp_path / "published"

    with PublicationLock(root, "exclusive_writer"):
        with pytest.raises(PublicationBusyError):
            with PublicationLock(root, "exclusive_writer"):
                pass


def _publication_bundle(tmp_path: Path, *, value: str):
    staged = tmp_path / f"staged-{value}"
    (staged / "translated").mkdir(parents=True)
    (staged / "translated" / "page_001.png").write_bytes(value.encode("utf-8"))
    (staged / "project.json").write_text(json.dumps({"value": value}), encoding="utf-8")
    (staged / "publication_receipt.json").write_text(
        json.dumps({"generation": value}), encoding="utf-8"
    )
    return SimpleNamespace(
        runtime_staging_root=staged,
        run_id="run-a",
        execution_id=f"execution-{value}",
        bundle_sha256=("a" if value == "old" else "b") * 64,
    )


def test_chapter_publication_promotes_receipt_last_and_replaces_whole_tree(tmp_path):
    public = tmp_path / "published"
    old = _publication_bundle(tmp_path, value="old")
    first = PublicationTransaction.for_bundle(public, old)
    first.stage(old)
    first.commit(old)
    new = _publication_bundle(tmp_path, value="new")
    second = PublicationTransaction.for_bundle(public, new)
    second.stage(new)
    second.commit(new)

    assert (public / "translated" / "page_001.png").read_bytes() == b"new"
    assert json.loads((public / "publication_receipt.json").read_text(encoding="utf-8")) == {
        "generation": "new"
    }


def test_chapter_publication_failure_rolls_back_whole_tree(tmp_path):
    public = tmp_path / "published"
    old = _publication_bundle(tmp_path, value="old")
    first = PublicationTransaction.for_bundle(public, old)
    first.stage(old)
    first.commit(old)
    new = _publication_bundle(tmp_path, value="new")
    transaction = PublicationTransaction.for_bundle(public, new)
    transaction.stage(new)
    transaction.fault_phase = "after_promote:project.json"

    with pytest.raises(RuntimeError, match="fault injection"):
        transaction.commit(new)

    assert (public / "translated" / "page_001.png").read_bytes() == b"old"
    assert json.loads((public / "publication_receipt.json").read_text(encoding="utf-8")) == {
        "generation": "old"
    }


def test_recover_chapter_publication_restores_interrupted_journal(tmp_path):
    public = tmp_path / "published"
    old = _publication_bundle(tmp_path, value="old")
    first = PublicationTransaction.for_bundle(public, old)
    first.stage(old)
    first.commit(old)
    new = _publication_bundle(tmp_path, value="new")
    interrupted = PublicationTransaction.for_bundle(public, new)
    interrupted.stage(new)
    interrupted.fault_phase = "after_promote:project.json"

    def fail_rollback(_journal):
        raise PublicationRecoveryError("simulated restore outage")

    interrupted._rollback_operations = fail_rollback
    with pytest.raises(PublicationRecoveryError, match="restore outage"):
        interrupted.commit(new)
    assert interrupted.journal_path.exists()

    recover_chapter_publication(public)
    recover_chapter_publication(public)

    assert (public / "translated" / "page_001.png").read_bytes() == b"old"
    assert json.loads((public / "publication_receipt.json").read_text(encoding="utf-8")) == {
        "generation": "old"
    }
