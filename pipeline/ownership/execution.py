"""Persisted page-execution artifacts and transactional evidence shells."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .hash_contract import (
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_page_sha256,
    sha256_bytes,
    sha256_file,
    sha256_text,
)


ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION = 1
PAGE_EXECUTION_EVIDENCE_SCHEMA_VERSION = 1
PAGE_EXECUTION_EVIDENCE_REF_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PageArtifactIntegrityError(ValueError):
    """Raised before exposing bytes whose path, identity, or hashes diverge."""


class PageNotTerminalError(ValueError):
    """Raised when candidate-only evidence reaches an export boundary."""


class AtomicReplacementError(ValueError):
    """Raised before a cleanup/target pair can create an owner commit."""


class TerminalVerificationIdentityError(PageArtifactIntegrityError):
    """Raised when terminal evidence does not belong to persisted candidate pixels."""


class TerminalVerificationInfrastructureError(RuntimeError):
    """Raised when a fresh physical terminal OCR attempt is unavailable."""


class SourceResidualStillMaterialError(ValueError):
    """Raised when source-language support remains materially visible."""


class TargetMaterializationError(ValueError):
    """Raised when the translated glyph patch is not materially present."""


class PageCandidateRecoveryError(PageArtifactIntegrityError):
    """Raised before recovery can touch an untrusted page transaction path."""


def canonical_glyph_patch_sha256(glyph_patch: Any) -> str:
    if glyph_patch is None:
        raise AtomicReplacementError("target glyph patch is missing")
    paint_mask = getattr(glyph_patch, "paint_mask", None)
    if paint_mask is None:
        paint_mask = getattr(glyph_patch, "glyph_mask", None)
    paint = np.asarray(paint_mask)
    result = np.asarray(getattr(glyph_patch, "result_rgb", None))
    if paint.ndim != 2 or result.ndim != 3 or result.shape[2] != 3:
        raise AtomicReplacementError("target glyph patch raster contract is invalid")
    payload = {
        "schema_version": 1,
        "owner_id": str(getattr(glyph_patch, "owner_id", "")),
        "page_id": str(getattr(glyph_patch, "page_id", "")),
        "coordinate_space": str(getattr(glyph_patch, "coordinate_space", "")),
        "before_sha256": str(getattr(glyph_patch, "before_sha256", "")),
        "after_sha256": str(getattr(glyph_patch, "after_sha256", "")),
        "glyph_mask_sha256": str(getattr(glyph_patch, "glyph_mask_sha256", "")),
        "paint_mask_sha256": str(getattr(glyph_patch, "paint_mask_sha256", "")),
        "component_geometry_sha256": str(
            getattr(glyph_patch, "component_geometry_sha256", "")
        ),
        "text_execution_authority_sha256": str(
            getattr(glyph_patch, "text_execution_authority_sha256", "")
        ),
        "delivery_contract_sha256": str(
            getattr(getattr(glyph_patch, "delivery_contract", None), "contract_sha256", "")
        ),
        "style_contract_sha256": str(
            getattr(getattr(glyph_patch, "style_raster_contract", None), "contract_sha256", "")
        ),
        "paint_shape": list(paint.shape),
        "paint_bytes_sha256": sha256_bytes(np.ascontiguousarray(paint).tobytes()),
        "result_pixel_sha256": canonical_page_sha256(result),
    }
    return canonical_json_sha256(payload)


@dataclass(frozen=True)
class OwnerReplacementOutcome:
    commit: Any
    materialization: Any
    final_page: Any
    final_pixel_sha256: str

    def __post_init__(self) -> None:
        pixels = np.ascontiguousarray(np.asarray(self.final_page, dtype=np.uint8)).copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "final_page", pixels)


@dataclass(frozen=True)
class OwnerReplacementAttemptResult:
    status: str
    outcome: OwnerReplacementOutcome | None
    repair_request: Any | None
    final_page: Any | None


def build_owner_repair_request(
    transaction: OwnerReplacementTransaction,
    *,
    failed_stage: str,
    reason: str,
    evidence_ids=(),
    next_strategy: str,
    issue_id: str | None = None,
) -> Any:
    from .model import OwnerRepairRequest

    if failed_stage == "final_qa" and not issue_id:
        raise AtomicReplacementError("final QA repair request requires issue identity")
    evidence = tuple(str(value) for value in evidence_ids)
    request_id = canonical_json_sha256(
        {
            "run_id": transaction.translation.run_id,
            "execution_id": transaction.execution_id,
            "page_id": transaction.translation.page_id,
            "page_source_sha256": transaction.translation.page_source_sha256,
            "owner_id": transaction.translation.owner_id,
            "failed_stage": str(failed_stage),
            "reason": str(reason),
            "evidence_ids": list(evidence),
            "next_strategy": str(next_strategy),
            "issue_id": issue_id,
        }
    )
    return OwnerRepairRequest.build(
        request_id=request_id,
        issue_id=issue_id,
        run_id=transaction.translation.run_id,
        execution_id=transaction.execution_id,
        page_id=transaction.translation.page_id,
        page_source_sha256=transaction.translation.page_source_sha256,
        owner_id=transaction.translation.owner_id,
        original_page_sha256=transaction.original_pixel_sha256,
        failed_stage=str(failed_stage),
        reason=str(reason),
        evidence_ids=evidence,
        next_strategy=str(next_strategy),
    )


def execute_owner_replacement(
    transaction: OwnerReplacementTransaction,
    cleanup: Any,
    glyph_patch: Any,
    *,
    failed_stage: str,
    failure_reason: str,
    evidence_ids=(),
    next_strategy: str,
    issue_id: str | None = None,
) -> OwnerReplacementAttemptResult:
    try:
        outcome = transaction.commit(cleanup, glyph_patch)
    except AtomicReplacementError:
        repair = build_owner_repair_request(
            transaction,
            failed_stage=failed_stage,
            reason=failure_reason,
            evidence_ids=evidence_ids,
            next_strategy=next_strategy,
            issue_id=issue_id,
        )
        return OwnerReplacementAttemptResult(
            status="repair_pending",
            outcome=None,
            repair_request=repair,
            final_page=None,
        )
    return OwnerReplacementAttemptResult(
        status="committed",
        outcome=outcome,
        repair_request=None,
        final_page=outcome.final_page,
    )


def bind_repair_ladder_to_page_result(page_result: Any, ladder_result: Any) -> Any:
    """Attach the canonical repair journal without promoting runtime pixels."""

    from strip.page_pipeline import PageExecutionResult

    requests_by_id = {
        item.request_id: item
        for item in (*page_result.repair_requests, *ladder_result.repair_requests)
    }
    attempts_by_id = {
        item.attempt_id: item
        for item in (*page_result.repair_history, *ladder_result.attempts)
    }
    status = (
        "repair_pending"
        if ladder_result.status == "repair_pending"
        else "candidate_ready"
    )
    return PageExecutionResult.build_from(
        page_result,
        repair_requests=tuple(
            sorted(requests_by_id.values(), key=lambda item: item.request_id)
        ),
        repair_history=tuple(
            sorted(attempts_by_id.values(), key=lambda item: item.attempt_id)
        ),
        repair_budget_policy_sha256=ladder_result.repair_budget_policy_sha256,
        status=status,
        final_page=None,
        terminal_proof=None,
    )


@dataclass(frozen=True)
class PageCompositionCandidate:
    final_page: Any
    base_pixel_sha256: str
    final_pixel_sha256: str
    commit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        pixels = np.ascontiguousarray(np.asarray(self.final_page, dtype=np.uint8)).copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "final_page", pixels)


@dataclass(frozen=True)
class OwnerReplacementTransaction:
    original_rgb: Any
    mutation: Any
    translation: Any
    execution_id: str
    original_pixel_sha256: str
    atomic_options: Mapping[str, Any]

    @classmethod
    def build(cls, **values: Any) -> "OwnerReplacementTransaction":
        original = np.asarray(values["original_rgb"])
        mutation = values["mutation"]
        translation = values["translation"]
        if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
            raise AtomicReplacementError("owner replacement original must be uint8 RGB")
        if (
            getattr(mutation, "owner_id", None) != getattr(translation, "owner_id", None)
            or getattr(mutation, "page_id", None) != getattr(translation, "page_id", None)
        ):
            raise AtomicReplacementError("cleanup and translation owner identity mismatch")
        frozen = np.ascontiguousarray(original).copy()
        frozen.setflags(write=False)
        return cls(
            original_rgb=frozen,
            mutation=mutation,
            translation=translation,
            execution_id=str(values["execution_id"]),
            original_pixel_sha256=canonical_page_sha256(original),
            atomic_options=dict(values.get("atomic_options") or {}),
        )

    def commit(self, cleanup: Any, glyph_patch: Any, *, overrides=None) -> Any:
        if cleanup is None or glyph_patch is None:
            raise AtomicReplacementError("cleanup and target glyph patch must commit together")
        if cleanup is not self.mutation:
            raise AtomicReplacementError("cleanup differs from sealed transaction mutation")
        if (
            getattr(cleanup, "owner_id", None) != self.translation.owner_id
            or getattr(glyph_patch, "owner_id", None) != self.translation.owner_id
            or getattr(cleanup, "page_id", None) != self.translation.page_id
            or getattr(glyph_patch, "page_id", None) != self.translation.page_id
        ):
            raise AtomicReplacementError("cleanup, glyph and translation identity mismatch")
        glyph_sha = canonical_glyph_patch_sha256(glyph_patch)
        expected = {
            "translation_binding_sha256": self.translation.translation_binding_sha256,
            "source_payload_sha256": self.translation.source_payload_sha256,
            "target_payload_sha256": self.translation.target_payload_sha256,
            "target_glyph_patch_sha256": glyph_sha,
        }
        supplied = expected | dict(overrides or {})
        if supplied != expected:
            raise AtomicReplacementError("stale translation or target hash override")
        from strip.process_bands import apply_atomic_owner_execution
        from .model import OwnerTargetMaterialization, bind_owner_execution_commit_identity

        raw_commit = apply_atomic_owner_execution(
            np.asarray(self.original_rgb), cleanup, glyph_patch, **dict(self.atomic_options)
        )
        if not raw_commit.committed:
            raise AtomicReplacementError(raw_commit.reason)
        final = np.asarray(raw_commit.result_rgb)
        final_sha = canonical_page_sha256(final)
        bound_payload = replace(
            raw_commit,
            before_sha256=self.original_pixel_sha256,
            after_sha256=final_sha,
            **expected,
        )
        commit = bind_owner_execution_commit_identity(
            bound_payload,
            run_id=self.translation.run_id,
            execution_id=self.execution_id,
            page_source_sha256=self.translation.page_source_sha256,
        )
        base_sha = canonical_page_sha256(np.asarray(cleanup.result_rgb))
        materialization_id = canonical_json_sha256(
            {
                "run_id": self.translation.run_id,
                "execution_id": self.execution_id,
                "page_id": self.translation.page_id,
                "owner_id": self.translation.owner_id,
                "translation_binding_sha256": self.translation.translation_binding_sha256,
                "target_glyph_patch_sha256": glyph_sha,
            }
        )
        materialization = OwnerTargetMaterialization.build(
            materialization_id=materialization_id,
            run_id=self.translation.run_id,
            execution_id=self.execution_id,
            page_id=self.translation.page_id,
            page_source_sha256=self.translation.page_source_sha256,
            owner_id=self.translation.owner_id,
            translation_binding_sha256=self.translation.translation_binding_sha256,
            source_payload_sha256=self.translation.source_payload_sha256,
            target_payload_sha256=self.translation.target_payload_sha256,
            target_glyph_patch_sha256=glyph_sha,
            glyph_mask_sha256=str(getattr(glyph_patch, "glyph_mask_sha256", "")),
            base_pixel_sha256=base_sha,
            result_pixel_sha256=final_sha,
        )
        return OwnerReplacementOutcome(
            commit=commit,
            materialization=materialization,
            final_page=final,
            final_pixel_sha256=final_sha,
        )


@dataclass(frozen=True)
class FrozenJSONSnapshot:
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "FrozenJSONSnapshot":
        encoded = canonical_json_bytes(dict(value))
        return cls(canonical_json_bytes=encoded, sha256=sha256_bytes(encoded))

    def read(self) -> dict[str, Any]:
        if sha256_bytes(self.canonical_json_bytes) != self.sha256:
            raise PageArtifactIntegrityError("frozen JSON snapshot hash mismatch")
        value = json.loads(self.canonical_json_bytes.decode("utf-8"))
        if not isinstance(value, dict):
            raise PageArtifactIntegrityError("frozen JSON snapshot root is not an object")
        return value


@dataclass(frozen=True)
class PageGeometrySnapshot:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    canonical_json_bytes: bytes
    sha256: str
    y_top: int
    y_bottom: int
    width: int
    height: int

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        execution_id: str,
        page_id: str,
        page_source_sha256: str,
        y_top: int,
        y_bottom: int,
        width: int,
        height: int,
    ) -> "PageGeometrySnapshot":
        _require_hash(page_source_sha256, "page_source_sha256")
        if not run_id or not execution_id or not page_id:
            raise PageArtifactIntegrityError("page geometry identity is incomplete")
        if min(width, height) <= 0 or y_top < 0 or y_bottom <= y_top or y_bottom - y_top != height:
            raise PageArtifactIntegrityError("page geometry dimensions are invalid")
        payload = {
            "run_id": run_id,
            "execution_id": execution_id,
            "page_id": page_id,
            "page_source_sha256": page_source_sha256,
            "y_top": int(y_top),
            "y_bottom": int(y_bottom),
            "width": int(width),
            "height": int(height),
        }
        encoded = canonical_json_bytes(payload)
        return cls(**payload, canonical_json_bytes=encoded, sha256=sha256_bytes(encoded))

    def read(self) -> dict[str, Any]:
        if sha256_bytes(self.canonical_json_bytes) != self.sha256:
            raise PageArtifactIntegrityError("page geometry snapshot hash mismatch")
        return json.loads(self.canonical_json_bytes.decode("utf-8"))


@dataclass(frozen=True)
class PageCompositionSnapshot:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    base_pixel_sha256: str
    final_pixel_sha256: str
    commit_ids: tuple[str, ...]
    translation_binding_sha256s: tuple[str, ...]
    source_payload_sha256s: tuple[str, ...]
    target_payload_sha256s: tuple[str, ...]
    target_glyph_patch_sha256s: tuple[str, ...]
    target_materialization_sha256s: tuple[str, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        execution_id: str,
        page_id: str,
        page_source_sha256: str,
        base_pixel_sha256: str,
        final_pixel_sha256: str,
        commits,
        materializations,
    ) -> "PageCompositionSnapshot":
        ordered_commits = tuple(commits)
        ordered_materializations = tuple(materializations)
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "execution_id": execution_id,
            "page_id": page_id,
            "page_source_sha256": page_source_sha256,
            "base_pixel_sha256": _require_hash(base_pixel_sha256, "base_pixel_sha256"),
            "final_pixel_sha256": _require_hash(final_pixel_sha256, "final_pixel_sha256"),
            "commit_ids": [str(item.commit_id) for item in ordered_commits],
            "translation_binding_sha256s": [
                str(item.translation_binding_sha256) for item in ordered_commits
            ],
            "source_payload_sha256s": [str(item.source_payload_sha256) for item in ordered_commits],
            "target_payload_sha256s": [str(item.target_payload_sha256) for item in ordered_commits],
            "target_glyph_patch_sha256s": [
                str(item.target_glyph_patch_sha256) for item in ordered_commits
            ],
            "target_materialization_sha256s": [
                str(item.materialization_sha256) for item in ordered_materializations
            ],
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            run_id=run_id,
            execution_id=execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            base_pixel_sha256=payload["base_pixel_sha256"],
            final_pixel_sha256=payload["final_pixel_sha256"],
            commit_ids=tuple(payload["commit_ids"]),
            translation_binding_sha256s=tuple(payload["translation_binding_sha256s"]),
            source_payload_sha256s=tuple(payload["source_payload_sha256s"]),
            target_payload_sha256s=tuple(payload["target_payload_sha256s"]),
            target_glyph_patch_sha256s=tuple(payload["target_glyph_patch_sha256s"]),
            target_materialization_sha256s=tuple(payload["target_materialization_sha256s"]),
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    def read(self) -> dict[str, Any]:
        if sha256_bytes(self.canonical_json_bytes) != self.sha256:
            raise PageArtifactIntegrityError("page composition snapshot hash mismatch")
        payload = json.loads(self.canonical_json_bytes.decode("utf-8"))
        if payload != self.to_dict(include_hash=False):
            raise PageArtifactIntegrityError("page composition snapshot canonical form mismatch")
        return payload

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "base_pixel_sha256": self.base_pixel_sha256,
            "final_pixel_sha256": self.final_pixel_sha256,
            "commit_ids": list(self.commit_ids),
            "translation_binding_sha256s": list(self.translation_binding_sha256s),
            "source_payload_sha256s": list(self.source_payload_sha256s),
            "target_payload_sha256s": list(self.target_payload_sha256s),
            "target_glyph_patch_sha256s": list(self.target_glyph_patch_sha256s),
            "target_materialization_sha256s": list(self.target_materialization_sha256s),
        }
        if include_hash:
            payload["sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PageCompositionSnapshot":
        expected = {
            "schema_version", "run_id", "execution_id", "page_id",
            "page_source_sha256", "base_pixel_sha256", "final_pixel_sha256",
            "commit_ids", "translation_binding_sha256s", "source_payload_sha256s",
            "target_payload_sha256s", "target_glyph_patch_sha256s",
            "target_materialization_sha256s", "sha256",
        }
        if set(payload) != expected or payload.get("schema_version") != 1:
            raise PageArtifactIntegrityError("page composition snapshot schema is invalid")
        rebuilt = cls.build(
            run_id=str(payload["run_id"]),
            execution_id=str(payload["execution_id"]),
            page_id=str(payload["page_id"]),
            page_source_sha256=str(payload["page_source_sha256"]),
            base_pixel_sha256=str(payload["base_pixel_sha256"]),
            final_pixel_sha256=str(payload["final_pixel_sha256"]),
            commits=tuple(
                type("CommitRef", (), {
                    "commit_id": commit_id,
                    "translation_binding_sha256": binding_sha,
                    "source_payload_sha256": source_sha,
                    "target_payload_sha256": target_sha,
                    "target_glyph_patch_sha256": glyph_sha,
                })()
                for commit_id, binding_sha, source_sha, target_sha, glyph_sha in zip(
                    payload["commit_ids"],
                    payload["translation_binding_sha256s"],
                    payload["source_payload_sha256s"],
                    payload["target_payload_sha256s"],
                    payload["target_glyph_patch_sha256s"],
                    strict=True,
                )
            ),
            materializations=tuple(
                type("MaterializationRef", (), {"materialization_sha256": value})()
                for value in payload["target_materialization_sha256s"]
            ),
        )
        if rebuilt.sha256 != payload["sha256"] or rebuilt.to_dict() != dict(payload):
            raise PageArtifactIntegrityError("page composition snapshot hash mismatch")
        return rebuilt


@dataclass(frozen=True)
class FinalPageSnapshot:
    lossless_png_bytes: bytes
    page_output_pixel_sha256: str
    final_file_sha256: str
    artifact_ref: "PersistedRGBImageArtifactRef"
    width: int
    height: int
    mode: str = "RGB"

    @classmethod
    def from_artifact(
        cls,
        artifact_ref: "PersistedRGBImageArtifactRef",
        generation_root: str | Path,
    ) -> "FinalPageSnapshot":
        pixels = artifact_ref.load_verified(generation_root)
        path = _resolve_asset(generation_root, artifact_ref.relative_path, must_exist=True)
        encoded = path.read_bytes()
        return cls(
            lossless_png_bytes=encoded,
            page_output_pixel_sha256=artifact_ref.pixel_sha256,
            final_file_sha256=artifact_ref.file_sha256,
            artifact_ref=artifact_ref,
            width=int(pixels.shape[1]),
            height=int(pixels.shape[0]),
        )

    def read_only_rgb(self) -> np.ndarray:
        if sha256_bytes(self.lossless_png_bytes) != self.final_file_sha256:
            raise PageArtifactIntegrityError("final page file hash mismatch")
        with Image.open(__import__("io").BytesIO(self.lossless_png_bytes)) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        if (
            rgb.shape != (self.height, self.width, 3)
            or self.mode != "RGB"
            or canonical_page_sha256(rgb) != self.page_output_pixel_sha256
            or self.page_output_pixel_sha256 != self.artifact_ref.pixel_sha256
        ):
            raise PageArtifactIntegrityError("final page pixel identity mismatch")
        rgb.setflags(write=False)
        return rgb


@dataclass(frozen=True)
class TerminalPixelProof:
    run_id: str
    execution_id: str
    page_id: str
    page_source_sha256: str
    final_page_pixel_sha256: str
    cleanup_base_sha256: str
    composition_sha256: str
    translation_binding_sha256s: tuple[str, ...]
    source_payload_sha256s: tuple[str, ...]
    target_payload_sha256s: tuple[str, ...]
    target_glyph_patch_sha256s: tuple[str, ...]
    target_materialization_sha256s: tuple[str, ...]
    final_replacement_verdict_sha256s: tuple[str, ...]
    repair_budget_policy_sha256: str
    replacement_verification_policy_sha256: str
    final_qa_probe_id: str
    fresh_ocr_invocation_id: str
    fresh_ocr_root_input_pixel_sha256: str
    fresh_ocr_attempt_ids: tuple[str, ...]
    fresh_ocr_attempt_chain_sha256: str
    fresh_ocr_payload_sha256: str
    source_support_removed: bool
    target_glyph_patch_applied: bool
    fresh_source_ocr_absent: bool
    coverage_complete: bool
    unowned_material_text_absent: bool
    proof_sha256: str

    @classmethod
    def build(cls, **values: Any) -> "TerminalPixelProof":
        normalized = dict(values)
        for key in (
            "translation_binding_sha256s", "source_payload_sha256s", "target_payload_sha256s",
            "target_glyph_patch_sha256s", "target_materialization_sha256s",
            "final_replacement_verdict_sha256s", "fresh_ocr_attempt_ids",
        ):
            normalized[key] = tuple(str(value) for value in normalized.get(key) or ())
        for key in (
            "page_source_sha256", "final_page_pixel_sha256", "cleanup_base_sha256",
            "composition_sha256", "repair_budget_policy_sha256",
            "replacement_verification_policy_sha256", "fresh_ocr_root_input_pixel_sha256",
            "fresh_ocr_attempt_chain_sha256", "fresh_ocr_payload_sha256",
        ):
            _require_hash(str(normalized.get(key) or ""), key)
        for key in (
            "translation_binding_sha256s", "source_payload_sha256s", "target_payload_sha256s",
            "target_glyph_patch_sha256s", "target_materialization_sha256s",
            "final_replacement_verdict_sha256s",
        ):
            for value in normalized[key]:
                _require_hash(value, key)
        if not all(str(normalized.get(key) or "").strip() for key in (
            "run_id", "execution_id", "page_id", "final_qa_probe_id", "fresh_ocr_invocation_id"
        )):
            raise PageArtifactIntegrityError("terminal proof identity is incomplete")
        payload = {key: normalized[key] for key in _terminal_proof_payload_fields()}
        proof_hash = canonical_json_sha256(_json_tuple_lists(payload))
        return cls(**payload, proof_sha256=proof_hash)

    @classmethod
    def build_from_persisted_candidate(
        cls,
        candidate: "FinalPageSnapshot",
        fresh_ocr: Any,
        *,
        final_qa_probe: Any,
        generation_root: str | Path,
        cleanup_base_sha256: str,
        composition_sha256: str,
        bindings=(),
        materializations=(),
        verdicts=(),
        repair_budget_policy_sha256: str,
        replacement_verification_policy_sha256: str,
        source_support_removed: bool,
        target_glyph_patch_applied: bool,
        fresh_source_ocr_absent: bool,
        coverage_complete: bool,
        unowned_material_text_absent: bool,
    ) -> "TerminalPixelProof":
        from .model import FinalQAProbe, FinalReplacementVerdict, OwnerTargetMaterialization
        from .ocr_contract import OCRInvocationResult

        if not isinstance(candidate, FinalPageSnapshot):
            raise TypeError("terminal proof requires a persisted FinalPageSnapshot")
        if not isinstance(fresh_ocr, OCRInvocationResult):
            raise TypeError("terminal proof requires one request-scoped OCR invocation")
        try:
            FinalQAProbe.from_dict(final_qa_probe.to_dict())
        except (AttributeError, TypeError, ValueError) as exc:
            raise TerminalVerificationIdentityError("final QA probe is invalid") from exc
        persisted_pixels = candidate.artifact_ref.load_verified(generation_root)
        snapshot_pixels = candidate.read_only_rgb()
        if not np.array_equal(persisted_pixels, snapshot_pixels):
            raise TerminalVerificationIdentityError("candidate snapshot differs from persisted pixels")
        candidate_pixel_sha = canonical_page_sha256(persisted_pixels)
        if (
            candidate_pixel_sha != candidate.page_output_pixel_sha256
            or candidate_pixel_sha != candidate.artifact_ref.pixel_sha256
            or sha256_bytes(candidate.lossless_png_bytes) != candidate.final_file_sha256
        ):
            raise TerminalVerificationIdentityError("persisted candidate identity is stale")

        request = fresh_ocr.request
        ref = candidate.artifact_ref
        expected_identity = (ref.run_id, ref.execution_id, ref.page_id)
        if (request.run_id, request.origin_execution_id, request.page_id) != expected_identity:
            raise TerminalVerificationIdentityError("fresh OCR crossed candidate execution")
        if request.root_input_pixel_sha256 != candidate_pixel_sha:
            raise TerminalVerificationIdentityError("fresh OCR root differs from candidate pixels")
        if (
            final_qa_probe.run_id,
            final_qa_probe.execution_id,
            final_qa_probe.page_id,
            final_qa_probe.page_source_sha256,
        ) != (
            request.run_id,
            request.origin_execution_id,
            request.page_id,
            request.page_source_sha256,
        ):
            raise TerminalVerificationIdentityError("final QA probe crossed OCR identity")
        if (
            final_qa_probe.input_origin != "redecoded_persisted_candidate"
            or final_qa_probe.candidate_file_sha256 != candidate.final_file_sha256
            or final_qa_probe.candidate_pixel_sha256 != candidate_pixel_sha
            or final_qa_probe.root_input_pixel_sha256 != candidate_pixel_sha
            or final_qa_probe.ocr_invocation_id != request.invocation_id
            or final_qa_probe.fresh_ocr_attempt_ids
            != tuple(item.attempt_id for item in fresh_ocr.attempts)
            or final_qa_probe.fresh_ocr_attempt_chain_sha256
            != fresh_ocr.attempt_chain_sha256
        ):
            raise TerminalVerificationIdentityError("final QA probe physical chain mismatch")
        if not final_qa_probe.observer_available or not final_qa_probe.coverage_complete:
            raise TerminalVerificationInfrastructureError(
                "terminal observer is unavailable or coverage is incomplete"
            )

        has_full_page = False
        for attempt in fresh_ocr.attempts:
            if not attempt.provider_called or attempt.cache_hit:
                raise TerminalVerificationInfrastructureError(
                    "terminal OCR attempt is not a fresh physical provider call"
                )
            if attempt.root_input_pixel_sha256 != candidate_pixel_sha:
                raise TerminalVerificationIdentityError("OCR attempt crossed candidate root")
            if attempt.parent_input_pixel_sha256 != candidate_pixel_sha:
                raise TerminalVerificationIdentityError("OCR attempt parent hash is stale")
            try:
                physical_input = attempt.transform_spec.replay(persisted_pixels)
            except (TypeError, ValueError) as exc:
                raise TerminalVerificationIdentityError("OCR transform cannot be replayed") from exc
            if (
                canonical_page_sha256(physical_input) != attempt.input_pixel_sha256
                or physical_input.shape
                != (attempt.input_height, attempt.input_width, 3)
                or attempt.input_mode != "RGB"
            ):
                raise TerminalVerificationIdentityError(
                    "OCR attempt hash or geometry differs from replayed pixels"
                )
            if attempt.variant_id == "full_page" and (
                attempt.input_bbox_page is None
                and attempt.input_pixel_sha256 == candidate_pixel_sha
            ):
                has_full_page = True
        if not has_full_page:
            raise TerminalVerificationInfrastructureError(
                "terminal OCR lacks an uncached physical full-page attempt"
            )

        ocr_text = "\n".join(
            record.text for record in (*fresh_ocr.observations, *fresh_ocr.full_page_lines)
        )
        ocr_payload_sha = sha256_text(ocr_text)
        if final_qa_probe.ocr_payload_sha256 != ocr_payload_sha:
            raise TerminalVerificationIdentityError("final QA OCR payload hash mismatch")

        ordered_bindings = tuple(sorted(bindings, key=lambda item: item.owner_id))
        ordered_materializations = tuple(
            sorted(materializations, key=lambda item: item.owner_id)
        )
        ordered_verdicts = tuple(sorted(verdicts, key=lambda item: item.owner_id))
        for materialization in ordered_materializations:
            OwnerTargetMaterialization.from_dict(materialization.to_dict())
        for verdict in ordered_verdicts:
            FinalReplacementVerdict.from_dict(verdict.to_dict())
        if not (
            len(ordered_bindings)
            == len(ordered_materializations)
            == len(ordered_verdicts)
        ):
            raise TargetMaterializationError(
                "terminal binding, materialization and verdict cardinality differs"
            )
        for binding, materialization, verdict in zip(
            ordered_bindings,
            ordered_materializations,
            ordered_verdicts,
            strict=True,
        ):
            if not (
                binding.owner_id == materialization.owner_id == verdict.owner_id
                and binding.translation_binding_sha256
                == materialization.translation_binding_sha256
                == verdict.translation_binding_sha256
                and binding.source_payload_sha256
                == materialization.source_payload_sha256
                == verdict.source_payload_sha256
                and binding.target_payload_sha256
                == materialization.target_payload_sha256
                == verdict.target_payload_sha256
                and materialization.target_glyph_patch_sha256
                == verdict.target_glyph_patch_sha256
                and materialization.materialization_sha256
                == verdict.target_materialization_sha256
                and verdict.replacement_verification_policy_sha256
                == replacement_verification_policy_sha256
                and verdict.status == "final_verified"
            ):
                raise TerminalVerificationIdentityError(
                    "terminal replacement hash chain mismatch"
                )
            if not verdict.source_support_removed:
                raise SourceResidualStillMaterialError(
                    "source support remains material after cleanup"
                )
            if not verdict.target_materialized:
                raise TargetMaterializationError("translated glyph patch is not material")

        if not source_support_removed or not fresh_source_ocr_absent:
            raise SourceResidualStillMaterialError("terminal source residual remains material")
        if not target_glyph_patch_applied:
            raise TargetMaterializationError("terminal target glyph patch is missing")
        if not coverage_complete or not unowned_material_text_absent:
            raise TerminalVerificationInfrastructureError(
                "terminal coverage contains unresolved material text"
            )
        return cls.build(
            run_id=request.run_id,
            execution_id=request.origin_execution_id,
            page_id=request.page_id,
            page_source_sha256=request.page_source_sha256,
            final_page_pixel_sha256=candidate_pixel_sha,
            cleanup_base_sha256=cleanup_base_sha256,
            composition_sha256=composition_sha256,
            translation_binding_sha256s=tuple(
                item.translation_binding_sha256 for item in ordered_bindings
            ),
            source_payload_sha256s=tuple(
                item.source_payload_sha256 for item in ordered_bindings
            ),
            target_payload_sha256s=tuple(
                item.target_payload_sha256 for item in ordered_bindings
            ),
            target_glyph_patch_sha256s=tuple(
                item.target_glyph_patch_sha256 for item in ordered_materializations
            ),
            target_materialization_sha256s=tuple(
                item.materialization_sha256 for item in ordered_materializations
            ),
            final_replacement_verdict_sha256s=tuple(
                item.verdict_sha256 for item in ordered_verdicts
            ),
            repair_budget_policy_sha256=repair_budget_policy_sha256,
            replacement_verification_policy_sha256=replacement_verification_policy_sha256,
            final_qa_probe_id=final_qa_probe.probe_id,
            fresh_ocr_invocation_id=request.invocation_id,
            fresh_ocr_root_input_pixel_sha256=candidate_pixel_sha,
            fresh_ocr_attempt_ids=tuple(item.attempt_id for item in fresh_ocr.attempts),
            fresh_ocr_attempt_chain_sha256=fresh_ocr.attempt_chain_sha256,
            fresh_ocr_payload_sha256=ocr_payload_sha,
            source_support_removed=True,
            target_glyph_patch_applied=True,
            fresh_source_ocr_absent=True,
            coverage_complete=True,
            unowned_material_text_absent=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_tuple_lists({key: getattr(self, key) for key in _terminal_proof_payload_fields()}) | {
            "proof_sha256": self.proof_sha256
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminalPixelProof":
        rebuilt = cls.build(**{key: payload.get(key) for key in _terminal_proof_payload_fields()})
        if rebuilt.proof_sha256 != payload.get("proof_sha256"):
            raise PageArtifactIntegrityError("terminal proof hash mismatch")
        return rebuilt


def build_final_qa_probe(
    candidate: FinalPageSnapshot,
    fresh_ocr: Any,
    *,
    generation_root: str | Path,
    issues=(),
) -> Any:
    """Build the persisted-candidate QA probe record from one OCR invocation."""

    from .model import FinalQAProbe, LanguageResidualIssue
    from .ocr_contract import OCRInvocationResult

    if not isinstance(candidate, FinalPageSnapshot) or not isinstance(
        fresh_ocr, OCRInvocationResult
    ):
        raise TypeError("final QA probe requires persisted pixels and OCR invocation")
    pixels = candidate.artifact_ref.load_verified(generation_root)
    pixel_sha = canonical_page_sha256(pixels)
    if pixel_sha != candidate.page_output_pixel_sha256:
        raise TerminalVerificationIdentityError("QA candidate pixels are stale")
    if fresh_ocr.root_input_pixel_sha256 != pixel_sha:
        raise TerminalVerificationIdentityError("QA invocation root differs from candidate")
    normalized_issues = tuple(issues)
    for issue in normalized_issues:
        LanguageResidualIssue.from_dict(issue.to_dict())
        if fresh_ocr.invocation_id not in issue.invocation_ids:
            raise TerminalVerificationIdentityError(
                "QA issue does not reference its physical invocation"
            )
    ocr_text = "\n".join(
        record.text for record in (*fresh_ocr.observations, *fresh_ocr.full_page_lines)
    )
    probe_id = canonical_json_sha256(
        {
            "run_id": fresh_ocr.run_id,
            "execution_id": fresh_ocr.origin_execution_id,
            "page_id": fresh_ocr.page_id,
            "candidate_pixel_sha256": pixel_sha,
            "ocr_invocation_id": fresh_ocr.invocation_id,
            "issue_ids": [item.issue_id for item in normalized_issues],
        }
    )
    return FinalQAProbe.build(
        probe_id=probe_id,
        run_id=fresh_ocr.run_id,
        execution_id=fresh_ocr.origin_execution_id,
        page_id=fresh_ocr.page_id,
        page_source_sha256=fresh_ocr.page_source_sha256,
        input_origin="redecoded_persisted_candidate",
        candidate_file_sha256=candidate.final_file_sha256,
        candidate_pixel_sha256=pixel_sha,
        root_input_pixel_sha256=fresh_ocr.root_input_pixel_sha256,
        ocr_invocation_id=fresh_ocr.invocation_id,
        fresh_ocr_attempt_ids=tuple(item.attempt_id for item in fresh_ocr.attempts),
        fresh_ocr_attempt_chain_sha256=fresh_ocr.attempt_chain_sha256,
        ocr_payload_sha256=sha256_text(ocr_text),
        observer_available=True,
        coverage_complete=True,
        issue_ids=tuple(item.issue_id for item in normalized_issues),
    )


def _mask_contract_sha256(mask: Any) -> tuple[str, int]:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("replacement verification mask must be two-dimensional")
    binary = np.ascontiguousarray(array > 0, dtype=np.uint8)
    return (
        canonical_json_sha256(
            {
                "shape": list(binary.shape),
                "bytes_sha256": sha256_bytes(binary.tobytes()),
            }
        ),
        int(np.count_nonzero(binary)),
    )


def build_final_replacement_verdict(
    *,
    binding: Any,
    materialization: Any,
    execution_id: str,
    source_support_mask: Any,
    post_cleanup_residual_mask: Any,
    target_delta_mask: Any,
    target_alpha_mask: Any,
    target_contrast_score: float,
    target_clipped_pixel_count: int,
    target_within_safe_region: bool,
    replacement_verification_policy_sha256: str,
) -> Any:
    """Build one fail-closed owner replacement verdict from material pixel evidence."""

    from .model import FinalReplacementVerdict, OwnerTargetMaterialization

    OwnerTargetMaterialization.from_dict(materialization.to_dict())
    if not (
        binding.owner_id == materialization.owner_id
        and binding.translation_binding_sha256
        == materialization.translation_binding_sha256
    ):
        raise TerminalVerificationIdentityError("verdict binding/materialization mismatch")
    source_support_sha, source_support_count = _mask_contract_sha256(source_support_mask)
    residual_sha, residual_count = _mask_contract_sha256(post_cleanup_residual_mask)
    target_delta_sha, target_delta_count = _mask_contract_sha256(target_delta_mask)
    _, target_alpha_count = _mask_contract_sha256(target_alpha_mask)
    source_removed = source_support_count > 0 and residual_count == 0
    target_materialized = (
        target_delta_count > 0
        and target_alpha_count > 0
        and float(target_contrast_score) > 0.0
        and int(target_clipped_pixel_count) == 0
        and bool(target_within_safe_region)
    )
    if not source_removed:
        raise SourceResidualStillMaterialError("source residual mask remains material")
    if not target_materialized:
        raise TargetMaterializationError("target glyph evidence is not material and contained")
    verdict_id = canonical_json_sha256(
        {
            "translation_binding_sha256": binding.translation_binding_sha256,
            "target_materialization_sha256": materialization.materialization_sha256,
            "source_support_sha256": source_support_sha,
            "target_delta_mask_sha256": target_delta_sha,
            "replacement_verification_policy_sha256": replacement_verification_policy_sha256,
        }
    )
    return FinalReplacementVerdict.build(
        verdict_id=verdict_id,
        run_id=binding.run_id,
        execution_id=execution_id,
        page_id=binding.page_id,
        page_source_sha256=binding.page_source_sha256,
        owner_id=binding.owner_id,
        translation_binding_sha256=binding.translation_binding_sha256,
        source_payload_sha256=binding.source_payload_sha256,
        target_payload_sha256=binding.target_payload_sha256,
        target_glyph_patch_sha256=materialization.target_glyph_patch_sha256,
        target_materialization_sha256=materialization.materialization_sha256,
        source_support_sha256=source_support_sha,
        post_cleanup_residual_mask_sha256=residual_sha,
        source_residual_material_pixel_count=residual_count,
        target_delta_mask_sha256=target_delta_sha,
        target_alpha_material_pixel_count=target_alpha_count,
        target_contrast_score=float(target_contrast_score),
        target_clipped_pixel_count=int(target_clipped_pixel_count),
        target_within_safe_region=bool(target_within_safe_region),
        source_support_removed=True,
        target_materialized=True,
        replacement_verification_policy_sha256=replacement_verification_policy_sha256,
        status="final_verified",
    )


@dataclass(frozen=True)
class PageExecutionEvidenceSnapshot:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    replay_source_page_evidence_sha256: str | None
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_content_semantic_sha256: str
    artifact_store_id: str
    generation_id: str
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, result: Any) -> "PageExecutionEvidenceSnapshot":
        payload = result.to_canonical_dict()
        encoded = canonical_json_bytes(payload)
        original_ref = getattr(result.request.original_page, "artifact_ref", None)
        if original_ref is None:
            raise PageArtifactIntegrityError("page evidence requires persisted original artifact")
        final_ref = getattr(getattr(result, "final_page", None), "artifact_ref", None)
        authority_ref = final_ref or original_ref
        return cls(
            schema_version=PAGE_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            run_id=result.request.run_id,
            execution_id=result.request.execution_id,
            replay_of_execution_id=result.request.replay_of_execution_id,
            replay_source_page_evidence_sha256=payload.get("replay_source_page_evidence_sha256"),
            page_id=result.page_id,
            page_source_sha256=result.request.page_source_sha256,
            page_result_sha256=result.result_sha256,
            page_content_semantic_sha256=result.page_content_semantic_sha256,
            artifact_store_id=authority_ref.artifact_store_id,
            generation_id=authority_ref.generation_id,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    @classmethod
    def read_verified(
        cls,
        encoded: bytes,
        generation_root: str | Path,
        *,
        expected: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageArtifactIntegrityError("page execution evidence JSON is invalid") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != PAGE_EXECUTION_EVIDENCE_SCHEMA_VERSION:
            raise PageArtifactIntegrityError("page execution evidence schema is unsupported")
        from strip.page_pipeline import PageExecutionResult

        result = PageExecutionResult.from_canonical_dict(payload, generation_root=generation_root)
        rebuilt = cls.build(result)
        if rebuilt.canonical_json_bytes != encoded:
            raise PageArtifactIntegrityError("page execution evidence is not canonical")
        for key, value in dict(expected or {}).items():
            if getattr(rebuilt, key) != value:
                raise PageArtifactIntegrityError(f"page execution evidence {key} mismatch")
        return result


@dataclass(frozen=True)
class PageExecutionEvidenceRef:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    artifact_store_id: str
    generation_id: str
    page_generation_id: str
    current_pointer_relative_path: str
    current_pointer_file_sha256: str
    page_execution_evidence_relative_path: str
    page_execution_evidence_file_sha256: str
    page_execution_evidence_sha256: str
    page_result_sha256: str
    pointer_sha256: str
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, **values: Any) -> "PageExecutionEvidenceRef":
        normalized = dict(values)
        for key in ("current_pointer_relative_path", "page_execution_evidence_relative_path"):
            normalized[key] = _relative_path(str(normalized.get(key) or ""))
        for key in (
            "page_source_sha256", "current_pointer_file_sha256",
            "page_execution_evidence_file_sha256", "page_execution_evidence_sha256",
            "page_result_sha256", "pointer_sha256",
        ):
            _require_hash(str(normalized.get(key) or ""), key)
        normalized["schema_version"] = PAGE_EXECUTION_EVIDENCE_REF_SCHEMA_VERSION
        payload = {key: normalized.get(key) for key in _page_evidence_ref_payload_fields()}
        encoded = canonical_json_bytes(payload)
        return cls(**payload, canonical_json_bytes=encoded, sha256=sha256_bytes(encoded))

    def read_verified(self, private_execution_root: str | Path) -> Any:
        marker = ArtifactGenerationMarker.read_verified(private_execution_root)
        _require_marker_identity(self, marker)
        pointer_path = _resolve_asset(
            private_execution_root, self.current_pointer_relative_path, must_exist=True
        )
        pointer_bytes = pointer_path.read_bytes()
        if sha256_bytes(pointer_bytes) != self.current_pointer_file_sha256:
            raise PageArtifactIntegrityError("page current pointer file hash mismatch")
        try:
            pointer = json.loads(pointer_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageArtifactIntegrityError("page current pointer JSON is invalid") from exc
        supplied_pointer_hash = pointer.pop("pointer_sha256", None)
        if canonical_json_sha256(pointer) != supplied_pointer_hash or supplied_pointer_hash != self.pointer_sha256:
            raise PageArtifactIntegrityError("page current pointer hash mismatch")
        for key in _page_pointer_fields():
            if pointer.get(key) != getattr(self, key):
                raise PageArtifactIntegrityError(f"page current pointer {key} mismatch")
        evidence_path = _resolve_asset(
            private_execution_root, self.page_execution_evidence_relative_path, must_exist=True
        )
        evidence_bytes = evidence_path.read_bytes()
        if sha256_bytes(evidence_bytes) != self.page_execution_evidence_file_sha256:
            raise PageArtifactIntegrityError("page execution evidence file hash mismatch")
        return PageExecutionEvidenceSnapshot.read_verified(
            evidence_bytes,
            private_execution_root,
            expected={
                "run_id": self.run_id,
                "execution_id": self.execution_id,
                "page_id": self.page_id,
                "page_source_sha256": self.page_source_sha256,
                "artifact_store_id": self.artifact_store_id,
                "generation_id": self.generation_id,
                "sha256": self.page_execution_evidence_sha256,
                "page_result_sha256": self.page_result_sha256,
            },
        )


@dataclass
class PageCandidateTransaction:
    private_execution_root: Path
    run_id: str
    execution_id: str
    page_id: str
    artifact_store_id: str
    generation_id: str
    page_generation_id: str
    transaction_id: str
    _composition_original_rgb: Any = None
    _composition_commits: tuple[Any, ...] = ()

    @classmethod
    def from_original(
        cls,
        original_rgb: Any,
        *,
        commits=(),
        artifact_root: str | Path | None = None,
    ) -> "PageCandidateTransaction":
        return cls(
            private_execution_root=Path(artifact_root or "."),
            run_id="",
            execution_id="",
            page_id="",
            artifact_store_id="",
            generation_id="",
            page_generation_id="",
            transaction_id="",
            _composition_original_rgb=np.asarray(original_rgb).copy(),
            _composition_commits=tuple(commits),
        )

    def replace_owner_commit(self, commit: Any) -> "PageCandidateTransaction":
        owner_id = str(getattr(commit, "owner_id", "") or "")
        if not owner_id:
            raise AtomicReplacementError("replacement commit owner identity is missing")
        commits = tuple(
            item for item in self._composition_commits
            if str(getattr(item, "owner_id", "") or "") != owner_id
        ) + (commit,)
        return replace(self, _composition_commits=commits)

    def compose(self) -> Any:
        if self._composition_original_rgb is None:
            raise AtomicReplacementError("page compositor has no immutable original")
        original = np.asarray(self._composition_original_rgb)
        if original.dtype != np.uint8 or original.ndim != 3 or original.shape[2] != 3:
            raise AtomicReplacementError("page compositor original is invalid")
        ordered = tuple(
            sorted(
                self._composition_commits,
                key=lambda item: (str(getattr(item, "owner_id", "")), str(getattr(item, "commit_id", ""))),
            )
        )
        owner_ids = [str(getattr(item, "owner_id", "")) for item in ordered]
        if len(owner_ids) != len(set(owner_ids)):
            raise AtomicReplacementError("page compositor has duplicate owner commits")
        final = np.ascontiguousarray(original).copy()
        for commit in ordered:
            if not bool(getattr(commit, "committed", False)):
                raise AtomicReplacementError("page compositor rejects uncommitted owner")
            mutation = getattr(commit, "mutation", None)
            action = np.asarray(getattr(mutation, "action_mask", None))
            cleaned = np.asarray(getattr(mutation, "result_rgb", None))
            if action.shape != final.shape[:2] or cleaned.shape != final.shape:
                raise AtomicReplacementError("owner cleanup raster shape mismatch")
            final[action > 0] = cleaned[action > 0]
        for commit in ordered:
            patch = getattr(commit, "glyph_patch", None)
            paint = getattr(patch, "paint_mask", None)
            if paint is None:
                paint = getattr(patch, "glyph_mask", None)
            paint_array = np.asarray(paint)
            rendered = np.asarray(getattr(patch, "result_rgb", None))
            if paint_array.shape != final.shape[:2] or rendered.shape != final.shape:
                raise AtomicReplacementError("owner glyph raster shape mismatch")
            final[paint_array > 0] = rendered[paint_array > 0]
        final_sha = canonical_page_sha256(final)
        return PageCompositionCandidate(
            final_page=final,
            base_pixel_sha256=canonical_page_sha256(original),
            final_pixel_sha256=final_sha,
            commit_ids=tuple(str(getattr(item, "commit_id", "")) for item in ordered),
        )

    def persist_candidate(self, pixels: Any, *, attempt_id: str) -> FinalPageSnapshot:
        marker = ArtifactGenerationMarker.read_verified(self.private_execution_root)
        if (
            marker.run_id,
            marker.execution_id,
            marker.artifact_store_id,
            marker.generation_id,
        ) != (
            self.run_id,
            self.execution_id,
            self.artifact_store_id,
            self.generation_id,
        ):
            raise PageArtifactIntegrityError(
                "candidate transaction identity differs from generation marker"
            )
        safe_segment = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
        for label, value in (
            ("execution_id", self.execution_id),
            ("page_id", self.page_id),
            ("attempt_id", str(attempt_id)),
        ):
            if not safe_segment.fullmatch(str(value or "")):
                raise PageArtifactIntegrityError(f"candidate {label} is not path-safe")
        rgb = np.asarray(pixels)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise PageArtifactIntegrityError("candidate pixels must be HxWx3 uint8 RGB")
        relative = (
            f".staging/{self.execution_id}/{self.page_id}/{attempt_id}/candidate.png"
        )
        path = _resolve_asset(self.private_execution_root, relative, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise PageArtifactIntegrityError("immutable candidate attempt already exists")
        temporary = path.with_name("candidate.tmp.png")
        if temporary.exists() or temporary.is_symlink():
            raise PageArtifactIntegrityError("candidate temporary path is not clean")
        with temporary.open("wb") as handle:
            Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(
                handle,
                format="PNG",
                optimize=False,
                compress_level=6,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        ref = PersistedRGBImageArtifactRef.from_file(
            self.private_execution_root,
            relative,
            marker=marker,
            page_id=self.page_id,
            origin_execution_id=self.execution_id,
        )
        return FinalPageSnapshot.from_artifact(ref, self.private_execution_root)

    def commit_verified_generation(self, result: Any) -> PageExecutionEvidenceRef:
        if getattr(result, "status", None) != "final_verified":
            raise PageNotTerminalError("only final_verified pages can be persisted as current")
        marker = ArtifactGenerationMarker.read_verified(self.private_execution_root)
        if (
            marker.run_id, marker.execution_id, marker.artifact_store_id, marker.generation_id
        ) != (self.run_id, self.execution_id, self.artifact_store_id, self.generation_id):
            raise PageArtifactIntegrityError("page transaction identity differs from generation marker")
        if result.page_id != self.page_id or result.request.execution_id != self.execution_id:
            raise PageArtifactIntegrityError("page transaction received another page execution")
        result.request.original_page.artifact_ref.load_verified(self.private_execution_root)
        result.final_page.artifact_ref.load_verified(self.private_execution_root)
        safe_segment = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
        for label, value in (
            ("page_id", self.page_id),
            ("page_generation_id", self.page_generation_id),
            ("transaction_id", self.transaction_id),
        ):
            if not safe_segment.fullmatch(str(value or "")):
                raise PageArtifactIntegrityError(f"page transaction {label} is not path-safe")
        generation_dir_relative = (
            f".page-generations/{self.page_id}/{self.page_generation_id}"
        )
        generation_dir = _resolve_asset(
            self.private_execution_root, generation_dir_relative, must_exist=False
        )
        generation_dir.parent.mkdir(parents=True, exist_ok=True)
        if generation_dir.exists():
            raise PageArtifactIntegrityError("immutable page generation already exists")
        temporary_dir = generation_dir.with_name(
            f".{self.page_generation_id}.{self.transaction_id}.tmp"
        )
        if temporary_dir.exists() or temporary_dir.is_symlink():
            raise PageArtifactIntegrityError("page generation staging is not clean")
        temporary_dir.mkdir()
        final_relative = f"{generation_dir_relative}/final.png"
        final_path = temporary_dir / "final.png"
        final_bytes = bytes(result.final_page.lossless_png_bytes)
        with final_path.open("wb") as handle:
            handle.write(final_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        final_ref_payload = {
            "run_id": marker.run_id,
            "execution_id": marker.execution_id,
            "origin_execution_id": result.request.origin_execution_id,
            "source_artifact_ref_sha256": getattr(
                result.final_page.artifact_ref, "artifact_ref_sha256", None
            ),
            "page_id": self.page_id,
            "artifact_store_id": marker.artifact_store_id,
            "generation_id": marker.generation_id,
            "relative_path": final_relative,
            "file_sha256": sha256_bytes(final_bytes),
            "pixel_sha256": result.final_page.page_output_pixel_sha256,
            "width": result.final_page.width,
            "height": result.final_page.height,
            "mode": "RGB",
        }
        final_ref = PersistedRGBImageArtifactRef(
            **final_ref_payload,
            artifact_ref_sha256=canonical_json_sha256(final_ref_payload),
        )
        rebound_final = FinalPageSnapshot(
            lossless_png_bytes=final_bytes,
            page_output_pixel_sha256=result.final_page.page_output_pixel_sha256,
            final_file_sha256=sha256_bytes(final_bytes),
            artifact_ref=final_ref,
            width=result.final_page.width,
            height=result.final_page.height,
        )
        from strip.page_pipeline import PageExecutionResult

        rebound_result = PageExecutionResult.build_from(result, final_page=rebound_final)
        snapshot = PageExecutionEvidenceSnapshot.build(rebound_result)
        for name in ("execution_result.json", "page_execution_evidence.json"):
            output = temporary_dir / name
            with output.open("wb") as handle:
                handle.write(snapshot.canonical_json_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        temporary_dir.replace(generation_dir)
        self._fault_checkpoint("after_generation_rename")
        generation_relative = f"{generation_dir_relative}/page_execution_evidence.json"
        generation_path = _resolve_asset(
            self.private_execution_root, generation_relative, must_exist=True
        )
        evidence_file_sha = sha256_file(generation_path)
        pointer_relative = f".page-current/{self.page_id}/current.json"
        pointer_path = _resolve_asset(self.private_execution_root, pointer_relative, must_exist=False)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_payload = {
            "schema_version": PAGE_EXECUTION_EVIDENCE_REF_SCHEMA_VERSION,
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "replay_of_execution_id": result.request.replay_of_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": result.request.page_source_sha256,
            "artifact_store_id": self.artifact_store_id,
            "generation_id": self.generation_id,
            "page_generation_id": self.page_generation_id,
            "current_pointer_relative_path": pointer_relative,
            "page_execution_evidence_relative_path": generation_relative,
            "page_execution_evidence_file_sha256": evidence_file_sha,
            "page_execution_evidence_sha256": snapshot.sha256,
            "page_result_sha256": rebound_result.result_sha256,
        }
        pointer_sha = canonical_json_sha256(pointer_payload)
        pointer_bytes = canonical_json_bytes(pointer_payload | {"pointer_sha256": pointer_sha})
        pointer_temporary = pointer_path.with_name(f".{pointer_path.name}.{self.transaction_id}.tmp")
        if pointer_temporary.exists() or pointer_temporary.is_symlink():
            raise PageArtifactIntegrityError("page pointer staging is not clean")
        with pointer_temporary.open("wb") as handle:
            handle.write(pointer_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        self._fault_checkpoint("before_pointer_replace")
        pointer_temporary.replace(pointer_path)
        self._fault_checkpoint("after_pointer_replace")
        return PageExecutionEvidenceRef.build(
            **pointer_payload,
            current_pointer_file_sha256=sha256_bytes(pointer_bytes),
            pointer_sha256=pointer_sha,
        )

    def recover(self) -> None:
        """Current pointers are atomic; immutable orphan generations remain non-authoritative."""

    def _fault_checkpoint(self, phase: str) -> None:
        del phase


def _reject_reparse_path(path: Path, root: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if not current.exists():
            break
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise PageCandidateRecoveryError("page recovery path contains a reparse point")


def recover_page_candidate_transaction(
    private_execution_root: str | Path,
    *,
    page_id: str,
) -> PageExecutionEvidenceRef | None:
    """Reopen the one authoritative pointer without trusting journal paths."""

    try:
        root = Path(private_execution_root).resolve(strict=True)
        marker = ArtifactGenerationMarker.read_verified(root)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(page_id or "")):
            raise PageCandidateRecoveryError("page recovery identity is not path-safe")
        pointer_relative = f".page-current/{page_id}/current.json"
        pointer_path = _resolve_asset(root, pointer_relative, must_exist=False)
        _reject_reparse_path(pointer_path, root)
        if not pointer_path.exists():
            return None
        raw_pointer = pointer_path.read_bytes()
        pointer = json.loads(raw_pointer.decode("utf-8"))
        expected_fields = {*_page_pointer_fields(), "pointer_sha256"}
        if not isinstance(pointer, dict) or set(pointer) != expected_fields:
            raise PageCandidateRecoveryError("page current pointer schema is invalid")
        supplied_hash = str(pointer.pop("pointer_sha256") or "")
        if canonical_json_sha256(pointer) != supplied_hash:
            raise PageCandidateRecoveryError("page current pointer hash is invalid")
        if (
            pointer.get("run_id") != marker.run_id
            or pointer.get("execution_id") != marker.execution_id
            or pointer.get("artifact_store_id") != marker.artifact_store_id
            or pointer.get("generation_id") != marker.generation_id
            or pointer.get("page_id") != page_id
            or pointer.get("current_pointer_relative_path") != pointer_relative
        ):
            raise PageCandidateRecoveryError("page current pointer identity is stale")
        generation_id = str(pointer.get("page_generation_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", generation_id):
            raise PageCandidateRecoveryError("page generation identity is not path-safe")
        expected_evidence = (
            f".page-generations/{page_id}/{generation_id}/page_execution_evidence.json"
        )
        if pointer.get("page_execution_evidence_relative_path") != expected_evidence:
            raise PageCandidateRecoveryError("page evidence path is not canonical")
        evidence_path = _resolve_asset(root, expected_evidence, must_exist=True)
        _reject_reparse_path(evidence_path, root)
        ref = PageExecutionEvidenceRef.build(
            **pointer,
            current_pointer_file_sha256=sha256_bytes(raw_pointer),
            pointer_sha256=supplied_hash,
        )
        ref.read_verified(root)
        return ref
    except PageCandidateRecoveryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise PageCandidateRecoveryError("page candidate recovery journal is untrusted") from exc


def recover_owner_staging(
    private_execution_root: str | Path,
    *,
    run_id: str,
    execution_id: str,
) -> tuple[str, ...]:
    """Remove only orphan temporary candidate files for the exact active execution."""

    root = Path(private_execution_root).resolve(strict=True)
    marker = ArtifactGenerationMarker.read_verified(root)
    if marker.run_id != run_id or marker.execution_id != execution_id:
        raise PageCandidateRecoveryError("staging recovery crossed run execution identity")
    staging = _resolve_asset(root, f".staging/{execution_id}", must_exist=False)
    _reject_reparse_path(staging, root)
    if not staging.exists():
        return ()
    removed = []
    for temporary in staging.rglob("candidate.tmp.png"):
        _reject_reparse_path(temporary, root)
        if not temporary.is_file() or temporary.is_symlink():
            raise PageCandidateRecoveryError("candidate temporary entry is not a regular file")
        temporary.unlink()
        removed.append(temporary.relative_to(root).as_posix())
    return tuple(sorted(removed))


def _require_hash(value: str, field: str) -> str:
    normalized = str(value or "")
    if not _SHA256_RE.fullmatch(normalized):
        raise PageArtifactIntegrityError(f"{field} is not a lowercase SHA-256")
    return normalized


def _relative_path(value: str) -> str:
    raw = str(value or "")
    if not raw or raw.startswith("/") or "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        raise PageArtifactIntegrityError(f"artifact path is not relative and portable: {raw!r}")
    parsed = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in parsed.parts) or parsed.as_posix() != raw:
        raise PageArtifactIntegrityError(f"artifact path is not canonical: {raw!r}")
    return raw


def _resolve_asset(generation_root: str | Path, relative_path: str, *, must_exist: bool) -> Path:
    root = Path(generation_root).resolve(strict=True)
    relative = _relative_path(relative_path)
    lexical = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = lexical.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PageArtifactIntegrityError(f"artifact path escapes generation root: {relative}") from exc
    if must_exist and not resolved.is_file():
        raise PageArtifactIntegrityError(f"artifact is missing or not a file: {relative}")
    return resolved


def _validate_generation_tree_names(root: Path) -> None:
    seen: dict[str, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        key = unicodedata.normalize("NFC", relative).casefold()
        previous = seen.get(key)
        if previous is not None and previous != relative:
            raise PageArtifactIntegrityError(
                f"generation contains NFC/casefold path collision: {previous!r} and {relative!r}"
            )
        seen[key] = relative


@dataclass(frozen=True)
class ArtifactGenerationMarker:
    schema_version: int
    artifact_store_id: str
    generation_id: str
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def create(
        cls,
        *,
        artifact_store_id: str,
        generation_id: str,
        run_id: str,
        execution_id: str,
        replay_of_execution_id: str | None = None,
    ) -> "ArtifactGenerationMarker":
        if not all(str(value or "").strip() for value in (artifact_store_id, generation_id, run_id, execution_id)):
            raise PageArtifactIntegrityError("artifact generation identity is incomplete")
        payload = {
            "schema_version": ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION,
            "artifact_store_id": artifact_store_id,
            "generation_id": generation_id,
            "run_id": run_id,
            "execution_id": execution_id,
            "replay_of_execution_id": replay_of_execution_id,
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            schema_version=ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION,
            artifact_store_id=artifact_store_id,
            generation_id=generation_id,
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    def write(self, generation_root: str | Path) -> Path:
        root = Path(generation_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / "artifact_generation.json"
        payload = json.loads(self.canonical_json_bytes.decode("utf-8"))
        payload["marker_sha256"] = self.sha256
        path.write_bytes(canonical_json_bytes(payload))
        return path

    @classmethod
    def read_verified(cls, generation_root: str | Path) -> "ArtifactGenerationMarker":
        root = Path(generation_root).resolve(strict=True)
        path = root / "artifact_generation.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageArtifactIntegrityError("artifact generation marker is missing or invalid") from exc
        expected = {
            "schema_version", "artifact_store_id", "generation_id", "run_id",
            "execution_id", "replay_of_execution_id", "marker_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise PageArtifactIntegrityError("artifact generation marker fields are invalid")
        if payload["schema_version"] != ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION:
            raise PageArtifactIntegrityError("artifact generation marker schema is unsupported")
        marker = cls.create(
            artifact_store_id=str(payload["artifact_store_id"]),
            generation_id=str(payload["generation_id"]),
            run_id=str(payload["run_id"]),
            execution_id=str(payload["execution_id"]),
            replay_of_execution_id=(
                str(payload["replay_of_execution_id"])
                if payload["replay_of_execution_id"] is not None else None
            ),
        )
        if marker.sha256 != payload["marker_sha256"]:
            raise PageArtifactIntegrityError("artifact generation marker hash mismatch")
        _validate_generation_tree_names(root)
        return marker


@dataclass(frozen=True)
class PersistedAssetRef:
    run_id: str
    execution_id: str
    origin_execution_id: str
    source_artifact_ref_sha256: str | None
    page_id: str | None
    artifact_store_id: str
    generation_id: str
    relative_path: str
    file_sha256: str
    size_bytes: int
    media_type: str
    decoded_mode: str | None
    decoded_pixel_sha256: str | None
    width: int | None
    height: int | None
    artifact_ref_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _asset_ref_payload(self) | {"artifact_ref_sha256": self.artifact_ref_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedAssetRef":
        supplied_hash = str(payload.get("artifact_ref_sha256") or "")
        rebuilt = cls.build(**{key: payload.get(key) for key in _asset_ref_payload_fields()})
        if rebuilt.artifact_ref_sha256 != supplied_hash:
            raise PageArtifactIntegrityError("artifact ref hash mismatch")
        return rebuilt

    @classmethod
    def build(cls, **values: Any) -> "PersistedAssetRef":
        normalized = dict(values)
        normalized["relative_path"] = _relative_path(str(normalized.get("relative_path") or ""))
        _require_hash(str(normalized.get("file_sha256") or ""), "file_sha256")
        source_hash = normalized.get("source_artifact_ref_sha256")
        if source_hash is not None:
            _require_hash(str(source_hash), "source_artifact_ref_sha256")
        pixel_hash = normalized.get("decoded_pixel_sha256")
        if pixel_hash is not None:
            _require_hash(str(pixel_hash), "decoded_pixel_sha256")
        if int(normalized.get("size_bytes") or 0) < 0:
            raise PageArtifactIntegrityError("artifact size is invalid")
        if not all(str(normalized.get(key) or "").strip() for key in ("run_id", "execution_id", "origin_execution_id", "artifact_store_id", "generation_id", "media_type")):
            raise PageArtifactIntegrityError("artifact ref identity is incomplete")
        payload = {
            key: normalized.get(key)
            for key in (
                "run_id", "execution_id", "origin_execution_id", "source_artifact_ref_sha256",
                "page_id", "artifact_store_id", "generation_id", "relative_path", "file_sha256",
                "size_bytes", "media_type", "decoded_mode", "decoded_pixel_sha256", "width", "height",
            )
        }
        return cls(**payload, artifact_ref_sha256=canonical_json_sha256(payload))

    @classmethod
    def from_file(
        cls,
        generation_root: str | Path,
        relative_path: str,
        *,
        marker: ArtifactGenerationMarker | None = None,
        page_id: str | None = None,
        media_type: str | None = None,
        origin_execution_id: str | None = None,
        source_artifact_ref_sha256: str | None = None,
    ) -> "PersistedAssetRef":
        marker = marker or ArtifactGenerationMarker.read_verified(generation_root)
        path = _resolve_asset(generation_root, relative_path, must_exist=True)
        return cls.build(
            run_id=marker.run_id,
            execution_id=marker.execution_id,
            origin_execution_id=origin_execution_id or marker.execution_id,
            source_artifact_ref_sha256=source_artifact_ref_sha256,
            page_id=page_id,
            artifact_store_id=marker.artifact_store_id,
            generation_id=marker.generation_id,
            relative_path=relative_path,
            file_sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            media_type=media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            decoded_mode=None,
            decoded_pixel_sha256=None,
            width=None,
            height=None,
        )

    def load_verified_bytes(self, generation_root: str | Path) -> bytes:
        marker = ArtifactGenerationMarker.read_verified(generation_root)
        _require_marker_identity(self, marker)
        path = _resolve_asset(generation_root, self.relative_path, must_exist=True)
        data = path.read_bytes()
        if len(data) != self.size_bytes or sha256_bytes(data) != self.file_sha256:
            raise PageArtifactIntegrityError("artifact byte hash or size mismatch")
        if canonical_json_sha256(_asset_ref_payload(self)) != self.artifact_ref_sha256:
            raise PageArtifactIntegrityError("artifact ref hash mismatch")
        return data


@dataclass(frozen=True)
class PersistedRGBImageArtifactRef:
    run_id: str
    execution_id: str
    origin_execution_id: str
    source_artifact_ref_sha256: str | None
    page_id: str
    artifact_store_id: str
    generation_id: str
    relative_path: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    mode: str
    artifact_ref_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _rgb_ref_payload(self) | {"artifact_ref_sha256": self.artifact_ref_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersistedRGBImageArtifactRef":
        values = {key: payload.get(key) for key in _rgb_ref_payload_fields()}
        values["relative_path"] = _relative_path(str(values["relative_path"] or ""))
        for field in ("file_sha256", "pixel_sha256"):
            _require_hash(str(values[field] or ""), field)
        if values["source_artifact_ref_sha256"] is not None:
            _require_hash(str(values["source_artifact_ref_sha256"]), "source_artifact_ref_sha256")
        if values["mode"] != "RGB" or int(values["width"] or 0) <= 0 or int(values["height"] or 0) <= 0:
            raise PageArtifactIntegrityError("RGB artifact metadata is invalid")
        normalized = {
            **values,
            "width": int(values["width"]),
            "height": int(values["height"]),
            "page_id": str(values["page_id"] or ""),
        }
        expected = canonical_json_sha256(normalized)
        if expected != payload.get("artifact_ref_sha256"):
            raise PageArtifactIntegrityError("RGB artifact ref hash mismatch")
        return cls(**normalized, artifact_ref_sha256=expected)

    @classmethod
    def from_file(
        cls,
        generation_root: str | Path,
        relative_path: str,
        *,
        marker: ArtifactGenerationMarker | None = None,
        page_id: str,
        origin_execution_id: str | None = None,
        source_artifact_ref_sha256: str | None = None,
    ) -> "PersistedRGBImageArtifactRef":
        marker = marker or ArtifactGenerationMarker.read_verified(generation_root)
        path = _resolve_asset(generation_root, relative_path, must_exist=True)
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        payload = {
            "run_id": marker.run_id,
            "execution_id": marker.execution_id,
            "origin_execution_id": origin_execution_id or marker.execution_id,
            "source_artifact_ref_sha256": source_artifact_ref_sha256,
            "page_id": page_id,
            "artifact_store_id": marker.artifact_store_id,
            "generation_id": marker.generation_id,
            "relative_path": _relative_path(relative_path),
            "file_sha256": sha256_file(path),
            "pixel_sha256": canonical_page_sha256(rgb),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "mode": "RGB",
        }
        return cls(**payload, artifact_ref_sha256=canonical_json_sha256(payload))

    def load_verified(self, generation_root: str | Path) -> np.ndarray:
        marker = ArtifactGenerationMarker.read_verified(generation_root)
        _require_marker_identity(self, marker)
        path = _resolve_asset(generation_root, self.relative_path, must_exist=True)
        if sha256_file(path) != self.file_sha256:
            raise PageArtifactIntegrityError("RGB artifact file hash mismatch")
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        if (
            rgb.shape != (self.height, self.width, 3)
            or self.mode != "RGB"
            or canonical_page_sha256(rgb) != self.pixel_sha256
        ):
            raise PageArtifactIntegrityError("RGB artifact decoded pixel identity mismatch")
        payload = _rgb_ref_payload(self)
        if canonical_json_sha256(payload) != self.artifact_ref_sha256:
            raise PageArtifactIntegrityError("RGB artifact ref hash mismatch")
        rgb.setflags(write=False)
        return rgb


def _require_marker_identity(ref: Any, marker: ArtifactGenerationMarker) -> None:
    if (
        ref.run_id != marker.run_id
        or ref.execution_id != marker.execution_id
        or ref.artifact_store_id != marker.artifact_store_id
        or ref.generation_id != marker.generation_id
    ):
        raise PageArtifactIntegrityError("artifact ref belongs to another generation")


def _asset_ref_payload(ref: PersistedAssetRef) -> dict[str, Any]:
    return {key: getattr(ref, key) for key in _asset_ref_payload_fields()}


def _asset_ref_payload_fields() -> tuple[str, ...]:
    return (
        "run_id", "execution_id", "origin_execution_id", "source_artifact_ref_sha256",
        "page_id", "artifact_store_id", "generation_id", "relative_path", "file_sha256",
        "size_bytes", "media_type", "decoded_mode", "decoded_pixel_sha256", "width", "height",
    )


def _rgb_ref_payload_fields() -> tuple[str, ...]:
    return (
        "run_id", "execution_id", "origin_execution_id", "source_artifact_ref_sha256",
        "page_id", "artifact_store_id", "generation_id", "relative_path", "file_sha256",
        "pixel_sha256", "width", "height", "mode",
    )


def _rgb_ref_payload(ref: PersistedRGBImageArtifactRef) -> dict[str, Any]:
    return {key: getattr(ref, key) for key in _rgb_ref_payload_fields()}


def _terminal_proof_payload_fields() -> tuple[str, ...]:
    return (
        "run_id", "execution_id", "page_id", "page_source_sha256",
        "final_page_pixel_sha256", "cleanup_base_sha256", "composition_sha256",
        "translation_binding_sha256s", "source_payload_sha256s", "target_payload_sha256s",
        "target_glyph_patch_sha256s", "target_materialization_sha256s",
        "final_replacement_verdict_sha256s", "repair_budget_policy_sha256",
        "replacement_verification_policy_sha256", "final_qa_probe_id",
        "fresh_ocr_invocation_id", "fresh_ocr_root_input_pixel_sha256",
        "fresh_ocr_attempt_ids", "fresh_ocr_attempt_chain_sha256", "fresh_ocr_payload_sha256",
        "source_support_removed", "target_glyph_patch_applied", "fresh_source_ocr_absent",
        "coverage_complete", "unowned_material_text_absent",
    )


def _page_pointer_fields() -> tuple[str, ...]:
    return (
        "schema_version", "run_id", "execution_id", "replay_of_execution_id", "page_id",
        "page_source_sha256", "artifact_store_id", "generation_id", "page_generation_id",
        "current_pointer_relative_path", "page_execution_evidence_relative_path",
        "page_execution_evidence_file_sha256", "page_execution_evidence_sha256",
        "page_result_sha256",
    )


def _page_evidence_ref_payload_fields() -> tuple[str, ...]:
    return (
        *_page_pointer_fields(),
        "current_pointer_file_sha256",
        "pointer_sha256",
    )


def _json_tuple_lists(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


__all__ = [
    "ARTIFACT_GENERATION_MARKER_SCHEMA_VERSION",
    "ArtifactGenerationMarker",
    "FinalPageSnapshot",
    "FrozenJSONSnapshot",
    "PageArtifactIntegrityError",
    "PageCandidateTransaction",
    "PageCandidateRecoveryError",
    "PageCompositionCandidate",
    "PageCompositionSnapshot",
    "PageExecutionEvidenceRef",
    "PageExecutionEvidenceSnapshot",
    "PageGeometrySnapshot",
    "PageNotTerminalError",
    "AtomicReplacementError",
    "OwnerReplacementAttemptResult",
    "OwnerReplacementOutcome",
    "OwnerReplacementTransaction",
    "PersistedAssetRef",
    "PersistedRGBImageArtifactRef",
    "SourceResidualStillMaterialError",
    "TargetMaterializationError",
    "TerminalPixelProof",
    "TerminalVerificationIdentityError",
    "TerminalVerificationInfrastructureError",
    "build_final_qa_probe",
    "build_final_replacement_verdict",
    "build_owner_repair_request",
    "bind_repair_ladder_to_page_result",
    "canonical_glyph_patch_sha256",
    "execute_owner_replacement",
    "recover_owner_staging",
    "recover_page_candidate_transaction",
]
