from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import json

import numpy as np
import pytest
from PIL import Image

from ownership.execution import (
    ArtifactGenerationMarker,
    FinalPageSnapshot,
    PageCandidateTransaction,
    PageCandidateRecoveryError,
    PersistedRGBImageArtifactRef,
    TerminalPixelProof,
    TerminalVerificationIdentityError,
    TerminalVerificationInfrastructureError,
    recover_page_candidate_transaction,
)
from ownership.hash_contract import canonical_json_sha256, canonical_page_sha256, sha256_text
from ownership.model import FinalQAProbe, LanguageResidualIssue
from ownership.ocr_contract import (
    OCRAttempt,
    OCRDiagnostics,
    OCRInvocationResult,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
)
from ownership.repair import PageRepairController
from qa.final_pixel_observer import DetectorOcrFinalPixelObserver


def _issue(kind: str, *, owner_id: str | None = "owner-a") -> LanguageResidualIssue:
    region_sha = canonical_json_sha256({"bbox": [1, 2, 30, 20], "owner": owner_id})
    issue_id = canonical_json_sha256(
        {"kind": kind, "owner_id": owner_id, "region_sha256": region_sha}
    )
    return LanguageResidualIssue.build(
        issue_id=issue_id,
        run_id="run-repair",
        execution_id="execution-repair",
        page_id="page-001",
        page_source_sha256="a" * 64,
        page_output_pixel_sha256="b" * 64,
        owner_id=owner_id,
        component_id=("component-a" if owner_id else None),
        container_id=("container-a" if owner_id else None),
        invocation_ids=("final-qa-1",),
        kind=kind,
        observed_text="SOURCE TEXT",
        observed_text_sha256=sha256_text("SOURCE TEXT"),
        source_binding_sha256=("c" * 64 if owner_id else None),
        region_sha256=region_sha,
        source_only_tokens=("source", "text"),
        repair_required=True,
    )


def test_page_repair_controller_routes_every_issue_without_fail_open():
    issues = (
        _issue("source_language_visible"),
        _issue("target_payload_missing"),
        _issue("cleanup_incomplete"),
        _issue("target_glyphs_missing"),
        _issue("mixed_language_overlay"),
        _issue("independently_detected_text_without_owner", owner_id=None),
    )

    actions = PageRepairController().plan(issues)

    assert {action.issue_id for action in actions} == {issue.issue_id for issue in issues}
    by_kind = {action.kind: action for action in actions}
    assert by_kind["source_language_visible"].start_strategy == "R0"
    assert by_kind["cleanup_incomplete"].action == "full_replacement"
    assert by_kind["target_payload_missing"].reuse_translation_binding is True
    assert by_kind["target_glyphs_missing"].action == "rerender_target"
    assert by_kind["mixed_language_overlay"].start_strategy == "R2"
    assert by_kind["independently_detected_text_without_owner"].reopen_coverage is True


def test_page_repair_controller_deduplicates_same_canonical_issue():
    issue = _issue("source_language_visible")
    actions = PageRepairController().plan((issue, issue))
    assert len(actions) == 1


def test_page_repair_controller_rejects_cross_execution_issue_set():
    issue = _issue("source_language_visible")
    payload = issue.to_dict()
    payload.pop("issue_sha256")
    payload["execution_id"] = "other-execution"
    other = LanguageResidualIssue.build(**payload)

    try:
        PageRepairController().plan((issue, other))
    except ValueError as exc:
        assert "execution" in str(exc)
    else:
        raise AssertionError("cross-execution issue set was accepted")


def _candidate(tmp_path: Path):
    root = tmp_path / "private"
    root.mkdir()
    marker = ArtifactGenerationMarker.create(
        artifact_store_id="store-terminal",
        generation_id="generation-terminal",
        run_id="run-terminal",
        execution_id="execution-terminal",
    )
    marker.write(root)
    pixels = np.arange(16 * 22 * 3, dtype=np.uint8).reshape(16, 22, 3)
    path = root / "staging" / "candidate.png"
    path.parent.mkdir()
    Image.fromarray(pixels, "RGB").save(path)
    ref = PersistedRGBImageArtifactRef.from_file(
        root,
        "staging/candidate.png",
        marker=marker,
        page_id="page-terminal",
    )
    return root, FinalPageSnapshot.from_artifact(ref, root)


def _fresh_probe(candidate, *, variant_id="full_page", input_sha256=None):
    ref = candidate.artifact_ref
    root_sha = candidate.page_output_pixel_sha256
    request = OCRRequest(
        run_id=ref.run_id,
        origin_execution_id=ref.execution_id,
        page_id=ref.page_id,
        page_source_sha256="9" * 64,
        root_input_pixel_sha256=root_sha,
        invocation_id="terminal-invocation",
        provider_family="terminal-provider",
    )
    attempt = OCRAttempt(
        attempt_id="terminal-attempt",
        run_id=request.run_id,
        origin_execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        root_input_pixel_sha256=root_sha,
        invocation_id=request.invocation_id,
        provider_family=request.provider_family,
        variant_id=variant_id,
        input_pixel_sha256=input_sha256 or root_sha,
        parent_input_pixel_sha256=root_sha,
        input_bbox_page=None,
        input_kind="full_page",
        transform_spec=OCRTransformSpec.build((OCRTransformOperation(kind="identity"),)),
        input_width=candidate.width,
        input_height=candidate.height,
        input_mode="RGB",
        provider_called=True,
        cache_hit=False,
    )
    invocation = OCRInvocationResult.build(
        request=request,
        attempts=(attempt,),
        diagnostics=OCRDiagnostics("terminal-provider"),
    )
    probe = FinalQAProbe.build(
        probe_id="terminal-probe",
        run_id=request.run_id,
        execution_id=request.origin_execution_id,
        page_id=request.page_id,
        page_source_sha256=request.page_source_sha256,
        input_origin="redecoded_persisted_candidate",
        candidate_file_sha256=candidate.final_file_sha256,
        candidate_pixel_sha256=root_sha,
        root_input_pixel_sha256=root_sha,
        ocr_invocation_id=request.invocation_id,
        fresh_ocr_attempt_ids=(attempt.attempt_id,),
        fresh_ocr_attempt_chain_sha256=invocation.attempt_chain_sha256,
        ocr_payload_sha256=sha256_text(""),
        observer_available=True,
        coverage_complete=True,
        issue_ids=(),
    )
    return invocation, probe


def _terminal_kwargs(root, probe):
    return dict(
        final_qa_probe=probe,
        generation_root=root,
        cleanup_base_sha256="1" * 64,
        composition_sha256="2" * 64,
        repair_budget_policy_sha256="3" * 64,
        replacement_verification_policy_sha256="4" * 64,
        source_support_removed=True,
        target_glyph_patch_applied=True,
        fresh_source_ocr_absent=True,
        coverage_complete=True,
        unowned_material_text_absent=True,
    )


def test_terminal_proof_replays_physical_attempt_from_persisted_candidate(tmp_path):
    root, candidate = _candidate(tmp_path)
    invocation, probe = _fresh_probe(candidate)

    proof = TerminalPixelProof.build_from_persisted_candidate(
        candidate, invocation, **_terminal_kwargs(root, probe)
    )

    assert proof.final_page_pixel_sha256 == candidate.page_output_pixel_sha256
    assert proof.fresh_ocr_root_input_pixel_sha256 == candidate.page_output_pixel_sha256
    assert proof.final_qa_probe_id == probe.probe_id


def test_terminal_proof_rejects_attempt_hash_not_replayed_from_candidate(tmp_path):
    root, candidate = _candidate(tmp_path)
    invocation, probe = _fresh_probe(candidate, input_sha256="8" * 64)

    with pytest.raises(TerminalVerificationIdentityError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate, invocation, **_terminal_kwargs(root, probe)
        )


def test_terminal_proof_requires_uncached_physical_full_page_attempt(tmp_path):
    root, candidate = _candidate(tmp_path)
    invocation, probe = _fresh_probe(candidate, variant_id="gray")

    with pytest.raises(TerminalVerificationInfrastructureError):
        TerminalPixelProof.build_from_persisted_candidate(
            candidate, invocation, **_terminal_kwargs(root, probe)
        )


def test_final_observer_materializes_request_scoped_ocr_journal(tmp_path):
    from vision_stack import runtime

    pixels = np.full((18, 24, 3), 227, dtype=np.uint8)
    path = tmp_path / "candidate.png"
    Image.fromarray(pixels, "RGB").save(path)

    class EmptyDetector:
        def detect(self, _pixels):
            return []

    class EmptyModel:
        def ocr(self, image, det=True, rec=True, cls=False):
            del image, det, rec, cls
            return [[]]

    from vision_stack.ocr import OCREngine

    empty_engine = OCREngine.__new__(OCREngine)
    empty_engine._backend = "paddleocr"
    empty_engine._model = EmptyModel()

    with patch.object(runtime, "_get_ocr_engine", return_value=empty_engine):
        observation = DetectorOcrFinalPixelObserver(
            detector=EmptyDetector(), runtime=runtime
        ).observe(
            path,
            source_language="en",
            run_id="run-journal",
            execution_id="execution-journal",
            page_id="page-journal",
            page_source_sha256="7" * 64,
        )

    assert observation.ocr_request == observation.ocr_invocation.request
    assert observation.ocr_request.run_id == "run-journal"
    assert {attempt.variant_id for attempt in observation.ocr_invocation.attempts} == {"full_page"}
    assert all(attempt.qualifies_as_fresh_physical_inference for attempt in observation.ocr_invocation.attempts)


def test_page_candidate_transaction_persists_lossless_attempt_before_qa(tmp_path):
    root, marker_candidate = _candidate(tmp_path)
    marker = ArtifactGenerationMarker.read_verified(root)
    transaction = PageCandidateTransaction(
        private_execution_root=root,
        run_id=marker.run_id,
        execution_id=marker.execution_id,
        page_id=marker_candidate.artifact_ref.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id="page-generation-terminal",
        transaction_id="transaction-terminal",
    )
    pixels = 255 - marker_candidate.read_only_rgb()

    persisted = transaction.persist_candidate(pixels, attempt_id="attempt-r2")

    assert persisted.page_output_pixel_sha256 == canonical_page_sha256(pixels)
    assert persisted.artifact_ref.relative_path.endswith(
        "/attempt-r2/candidate.png"
    )
    np.testing.assert_array_equal(persisted.read_only_rgb(), pixels)


def test_page_candidate_recovery_rejects_traversal_before_external_access(tmp_path):
    root, _candidate_snapshot = _candidate(tmp_path)
    marker = ArtifactGenerationMarker.read_verified(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve", encoding="utf-8")
    pointer_path = root / ".page-current" / "page-terminal" / "current.json"
    pointer_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "run_id": marker.run_id,
        "execution_id": marker.execution_id,
        "replay_of_execution_id": None,
        "page_id": "page-terminal",
        "page_source_sha256": "9" * 64,
        "artifact_store_id": marker.artifact_store_id,
        "generation_id": marker.generation_id,
        "page_generation_id": "page-generation-terminal",
        "current_pointer_relative_path": ".page-current/page-terminal/current.json",
        "page_execution_evidence_relative_path": "../outside.txt",
        "page_execution_evidence_file_sha256": "a" * 64,
        "page_execution_evidence_sha256": "b" * 64,
        "page_result_sha256": "c" * 64,
    }
    payload["pointer_sha256"] = canonical_json_sha256(payload)
    pointer_path.write_text(json.dumps(payload), encoding="utf-8")
    outside_before = outside.read_bytes()

    with pytest.raises(PageCandidateRecoveryError):
        recover_page_candidate_transaction(root, page_id="page-terminal")

    assert outside.read_bytes() == outside_before
    assert pointer_path.exists()
