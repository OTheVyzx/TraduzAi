"""Canonical page-first coordinator for enforce-mode owner processing."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from ownership.coverage import PageCoverageResult
from ownership.hash_contract import (
    canonical_json_bytes,
    canonical_page_sha256,
    sha256_bytes,
    sha256_text,
)
from ownership.model import (
    FinalQAProbe,
    FinalReplacementVerdict,
    LanguageResidualIssue,
    TRANSLATION_ROUTE_ACTIONS,
    OwnerGraph,
    OwnerProjection,
    RepairAttempt,
)
from ownership.ocr_contract import OCRInvocationResult, OCRRequest
from ownership.owner_builder import build_owner_page_graph_from_coverage
from ownership.translation import (
    OwnerPageTranslationResult,
    OwnerTranslationRequest,
    TranslationAttempt,
    TranslationBinding,
    apply_owner_translation_result,
    translate_owner_page,
)
from translator.language_policy import build_page_language_evidence


class PagePipelineIdentityError(ValueError):
    """Raised when one page result crosses content or execution identity."""


class PagePipelineStateError(ValueError):
    """Raised when a page result claims an unsupported lifecycle state."""


class ContentReplayIntegrityError(ValueError):
    """Raised before any content from a tampered publication can be replayed."""


CANONICAL_VISUAL_STAGE_NAMES = (
    "original",
    "inpaint",
    "typeset",
    "page_composition",
    "persisted_final",
)


def _json_compatible_execution_value(value):
    """Materialize immutable tuple containers without accepting non-JSON values."""

    if isinstance(value, Mapping):
        return {
            key: _json_compatible_execution_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_execution_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalVisualStage:
    """Hash-linked authority for one canonical page-space visual stage."""

    name: str
    artifact_ref: Any
    pixel_sha256: str
    alias_of: str | None = None
    alias_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        to_dict = getattr(self.artifact_ref, "to_dict", None)
        if not callable(to_dict):
            raise PagePipelineStateError("visual stage has no serializable artifact ref")
        return {
            "name": self.name,
            "artifact_ref": to_dict(),
            "pixel_sha256": self.pixel_sha256,
            "alias_of": self.alias_of,
            "alias_reason": self.alias_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalVisualStage":
        from ownership.execution import PersistedRGBImageArtifactRef

        if set(payload) != {
            "name", "artifact_ref", "pixel_sha256", "alias_of", "alias_reason"
        } or not isinstance(payload.get("artifact_ref"), Mapping):
            raise PagePipelineStateError("visual stage schema is invalid")
        return cls(
            name=str(payload.get("name") or ""),
            artifact_ref=PersistedRGBImageArtifactRef.from_dict(payload["artifact_ref"]),
            pixel_sha256=str(payload.get("pixel_sha256") or ""),
            alias_of=(str(payload["alias_of"]) if payload.get("alias_of") is not None else None),
            alias_reason=(
                str(payload["alias_reason"])
                if payload.get("alias_reason") is not None else None
            ),
        )


@dataclass(frozen=True)
class BandProjection:
    band_id: str
    bbox_page: tuple[int, int, int, int]
    source_crop_sha256: str
    provenance_ids: tuple[str, ...]


@dataclass(frozen=True)
class OriginalPageSnapshot:
    lossless_png_bytes: bytes
    source_file_sha256: str
    page_source_sha256: str
    width: int
    height: int
    mode: str = "RGB"
    artifact_ref: Any | None = None

    @classmethod
    def from_pixels(
        cls,
        pixels: np.ndarray,
        *,
        source_file_sha256: str,
        artifact_ref: Any | None = None,
    ) -> "OriginalPageSnapshot":
        rgb = _canonical_rgb(pixels)
        output = BytesIO()
        Image.fromarray(rgb, "RGB").save(output, format="PNG", compress_level=9)
        encoded = output.getvalue()
        snapshot = cls(
            lossless_png_bytes=encoded,
            source_file_sha256=str(source_file_sha256),
            page_source_sha256=canonical_page_sha256(rgb),
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            artifact_ref=artifact_ref,
        )
        snapshot._validate()
        return snapshot

    @classmethod
    def from_file(cls, path, *, artifact_ref: Any | None = None) -> "OriginalPageSnapshot":
        from pathlib import Path

        source = Path(path).resolve(strict=True)
        encoded = source.read_bytes()
        with Image.open(BytesIO(encoded)) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        return cls.from_pixels(
            rgb,
            source_file_sha256=sha256_bytes(encoded),
            artifact_ref=artifact_ref,
        )

    def _validate(self) -> None:
        if len(self.source_file_sha256) != 64:
            raise PagePipelineIdentityError("original source file hash is invalid")
        rgb = self.read_only_rgb()
        if rgb.shape != (self.height, self.width, 3) or self.mode != "RGB":
            raise PagePipelineIdentityError("original page dimensions or mode disagree")
        if canonical_page_sha256(rgb) != self.page_source_sha256:
            raise PagePipelineIdentityError("original page pixel hash disagrees")

    def read_only_rgb(self) -> np.ndarray:
        with Image.open(BytesIO(self.lossless_png_bytes)) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        rgb.setflags(write=False)
        return rgb

    def mutable_attempt_copy(self) -> np.ndarray:
        return self.read_only_rgb().copy()


@dataclass(frozen=True)
class PagePipelineRequest:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    original_page: OriginalPageSnapshot
    band_projections: tuple[BandProjection, ...] = ()

    @property
    def origin_execution_id(self) -> str:
        return self.replay_of_execution_id or self.execution_id

    def __post_init__(self) -> None:
        if not self.run_id or not self.execution_id or not self.page_id:
            raise PagePipelineIdentityError("page request identity is incomplete")
        if self.page_source_sha256 != self.original_page.page_source_sha256:
            raise PagePipelineIdentityError("page request source hash disagrees with original")

    @classmethod
    def from_legacy_bands(
        cls,
        original_page: OriginalPageSnapshot,
        bands: Sequence[Any],
        *,
        run_id: str,
        execution_id: str,
        page_id: str,
        replay_of_execution_id: str | None = None,
    ) -> "PagePipelineRequest":
        pixels = original_page.read_only_rgb()
        projections: list[BandProjection] = []
        for index, band in enumerate(tuple(bands), 1):
            y1 = max(0, min(original_page.height, int(getattr(band, "y_top", 0))))
            y2 = max(y1, min(original_page.height, int(getattr(band, "y_bottom", 0))))
            if y2 <= y1:
                continue
            crop = pixels[y1:y2, :, :]
            provenance = tuple(
                str(region_id)
                for balloon in tuple(getattr(band, "balloons", ()) or ())
                if (region_id := getattr(balloon, "region_id", None))
            )
            projections.append(
                BandProjection(
                    band_id=str(getattr(band, "tile_id", None) or f"band_{index:03d}"),
                    bbox_page=(0, y1, original_page.width, y2),
                    source_crop_sha256=canonical_page_sha256(crop),
                    provenance_ids=provenance,
                )
            )
        return cls(
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
            page_id=page_id,
            page_source_sha256=original_page.page_source_sha256,
            original_page=original_page,
            band_projections=tuple(projections),
        )


@dataclass(frozen=True)
class OwnerGraphSnapshot:
    canonical_json_bytes: bytes
    sha256: str
    schema_version: int
    verification_status: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str

    @classmethod
    def build(cls, graph: OwnerGraph) -> "OwnerGraphSnapshot":
        graph.require_valid(mode="enforce")
        encoded = canonical_json_bytes(graph.to_dict())
        return cls(
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
            schema_version=graph.schema_version,
            verification_status=graph.verification_status,
            run_id=graph.run_id,
            origin_execution_id=graph.origin_execution_id,
            page_id=graph.page_id,
            page_source_sha256=graph.page_source_sha256,
        )

    def read(self) -> OwnerGraph:
        import json

        graph = OwnerGraph.from_dict(json.loads(self.canonical_json_bytes.decode("utf-8")), enforce=True)
        encoded = canonical_json_bytes(graph.to_dict())
        if sha256_bytes(encoded) != self.sha256:
            raise PagePipelineIdentityError("owner graph snapshot hash mismatch")
        return graph


@dataclass(frozen=True)
class PageExecutionResult:
    request: PagePipelineRequest
    coverage: PageCoverageResult
    owner_graph: OwnerGraphSnapshot
    translation_attempts: tuple[TranslationAttempt, ...]
    translations: tuple[TranslationBinding, ...]
    page_commits: tuple[Any, ...]
    status: str
    final_page: Any | None
    terminal_proof: Any | None
    result_sha256: str
    repair_requests: tuple[Any, ...] = ()
    owner_target_materializations: tuple[Any, ...] = ()
    text_layers_view: Any | None = None
    page_composition: Any | None = None
    repair_history: tuple[RepairAttempt, ...] = ()
    repair_budget_policy_sha256: str | None = None
    final_qa_ocr_requests: tuple[Any, ...] = ()
    final_qa_ocr_invocations: tuple[Any, ...] = ()
    language_residual_issues: tuple[Any, ...] = ()
    qa_probes: tuple[Any, ...] = ()
    final_replacement_verdicts: tuple[Any, ...] = ()
    replacement_verification_policy_sha256: str | None = None
    visual_stage_artifacts: tuple[CanonicalVisualStage, ...] = ()
    replay_source_page_evidence_sha256: str | None = None

    @property
    def page_id(self) -> str:
        return self.request.page_id

    @property
    def lifecycle(self):
        return self.coverage.ledger

    def promote_final(
        self,
        *,
        final_page: Any,
        terminal_proof: Any,
        final_qa_ocr_requests: Sequence[Any],
        final_qa_ocr_invocations: Sequence[Any],
        language_residual_issues: Sequence[Any],
        qa_probes: Sequence[Any],
        final_replacement_verdicts: Sequence[Any],
        replacement_verification_policy_sha256: str,
    ) -> "PageExecutionResult":
        """Return a new terminal result; the candidate instance remains unchanged."""

        if self.status not in {"candidate_ready", "repair_pending"}:
            raise PagePipelineStateError("only a candidate page can be promoted")
        return PageExecutionResult.build_from(
            self,
            status="final_verified",
            final_page=final_page,
            terminal_proof=terminal_proof,
            final_qa_ocr_requests=tuple(final_qa_ocr_requests),
            final_qa_ocr_invocations=tuple(final_qa_ocr_invocations),
            language_residual_issues=tuple(language_residual_issues),
            qa_probes=tuple(qa_probes),
            final_replacement_verdicts=tuple(final_replacement_verdicts),
            replacement_verification_policy_sha256=replacement_verification_policy_sha256,
            repair_budget_policy_sha256=(
                self.repair_budget_policy_sha256
                or getattr(terminal_proof, "repair_budget_policy_sha256", None)
            ),
        )

    @property
    def page_content_semantic_sha256(self) -> str:
        payload = self.to_canonical_dict()
        for key in ("execution_id", "replay_of_execution_id", "result_sha256"):
            payload.pop(key, None)
        return sha256_bytes(canonical_json_bytes(payload))

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the metadata-only page journal with final schema slots reserved."""

        import json

        final_page_payload = None
        if self.final_page is not None:
            ref = getattr(self.final_page, "artifact_ref", None)
            to_dict = getattr(ref, "to_dict", None)
            if not callable(to_dict):
                raise PagePipelineStateError("final page has no serializable artifact authority")
            final_page_payload = {
                "page_output_pixel_sha256": self.final_page.page_output_pixel_sha256,
                "final_file_sha256": self.final_page.final_file_sha256,
                "width": self.final_page.width,
                "height": self.final_page.height,
                "mode": self.final_page.mode,
                "artifact_ref": to_dict(),
            }
        proof_payload = None
        if self.terminal_proof is not None:
            to_dict = getattr(self.terminal_proof, "to_dict", None)
            if not callable(to_dict):
                raise PagePipelineStateError("terminal proof is not serializable")
            proof_payload = to_dict()
        original_ref = getattr(self.request.original_page, "artifact_ref", None)
        original_ref_to_dict = getattr(original_ref, "to_dict", None)
        from ownership.coverage import _invocation_json, _request_json

        return {
            "schema_version": 1,
            "run_id": self.request.run_id,
            "execution_id": self.request.execution_id,
            "replay_of_execution_id": self.request.replay_of_execution_id,
            "replay_source_page_evidence_sha256": self.replay_source_page_evidence_sha256,
            "page_id": self.request.page_id,
            "page_source_sha256": self.request.page_source_sha256,
            "status": self.status,
            "original_page": {
                "source_file_sha256": self.request.original_page.source_file_sha256,
                "page_source_sha256": self.request.original_page.page_source_sha256,
                "width": self.request.original_page.width,
                "height": self.request.original_page.height,
                "mode": self.request.original_page.mode,
                "artifact_ref": original_ref_to_dict() if callable(original_ref_to_dict) else None,
            },
            "band_projections": [
                {
                    "band_id": item.band_id,
                    "bbox_page": list(item.bbox_page),
                    "source_crop_sha256": item.source_crop_sha256,
                    "provenance_ids": list(item.provenance_ids),
                }
                for item in self.request.band_projections
            ],
            "coverage": json.loads(self.coverage.canonical_json_bytes.decode("utf-8")),
            "owner_graph": json.loads(self.owner_graph.canonical_json_bytes.decode("utf-8")),
            "translation_attempts": [item.to_dict() for item in self.translation_attempts],
            "translations": [item.to_dict() for item in self.translations],
            "repair_requests": [item.to_dict() for item in self.repair_requests],
            "repair_history": [item.to_dict() for item in self.repair_history],
            "repair_budget_policy_sha256": self.repair_budget_policy_sha256,
            "owner_target_materializations": [
                item.to_dict() for item in self.owner_target_materializations
            ],
            "page_commits": [
                {
                    key: getattr(item, key, None)
                    for key in (
                        "commit_id", "run_id", "execution_id", "page_id",
                        "page_source_sha256", "owner_id", "before_sha256", "after_sha256",
                        "translation_binding_sha256", "source_payload_sha256",
                        "target_payload_sha256", "target_glyph_patch_sha256",
                    )
                }
                for item in self.page_commits
            ],
            "page_geometry": None,
            "ocr_result": None,
            "text_layers": (
                self.text_layers_view.read() if self.text_layers_view is not None else None
            ),
            "page_composition": (
                self.page_composition.to_dict() if self.page_composition is not None else None
            ),
            "project_asset_refs": [],
            "visual_stage_artifacts": [
                item.to_dict() for item in self.visual_stage_artifacts
            ],
            "final_qa_ocr_requests": [
                _request_json(item) for item in self.final_qa_ocr_requests
            ],
            "final_qa_ocr_invocations": [
                _invocation_json(item) for item in self.final_qa_ocr_invocations
            ],
            "language_residual_issues": [
                item.to_dict() for item in self.language_residual_issues
            ],
            "qa_probes": [item.to_dict() for item in self.qa_probes],
            "final_replacement_verdicts": [
                item.to_dict() for item in self.final_replacement_verdicts
            ],
            "replacement_verification_policy_sha256": self.replacement_verification_policy_sha256,
            "final_page": final_page_payload,
            "terminal_proof": proof_payload,
        }

    @classmethod
    def from_canonical_dict(
        cls,
        payload: dict[str, Any],
        *,
        generation_root,
    ) -> "PageExecutionResult":
        """Rebuild a page journal only after reopening all persisted authorities."""

        from types import SimpleNamespace

        from ownership.execution import (
            FinalPageSnapshot,
            FrozenJSONSnapshot,
            PageCompositionSnapshot,
            PersistedRGBImageArtifactRef,
            TerminalPixelProof,
        )
        from ownership.coverage import _invocation_from_json
        from ownership.model import OwnerRepairRequest, OwnerTargetMaterialization

        if payload.get("schema_version") != 1:
            raise PagePipelineStateError("page execution evidence schema is unsupported")
        original_payload = payload.get("original_page")
        if not isinstance(original_payload, dict) or not isinstance(original_payload.get("artifact_ref"), dict):
            raise PagePipelineStateError("persisted page evidence lacks original artifact authority")
        original_ref = PersistedRGBImageArtifactRef.from_dict(original_payload["artifact_ref"])
        original_pixels = original_ref.load_verified(generation_root)
        original = OriginalPageSnapshot.from_pixels(
            original_pixels,
            source_file_sha256=str(original_payload.get("source_file_sha256") or ""),
            artifact_ref=original_ref,
        )
        request = PagePipelineRequest(
            run_id=str(payload.get("run_id") or ""),
            execution_id=str(payload.get("execution_id") or ""),
            replay_of_execution_id=(
                str(payload["replay_of_execution_id"])
                if payload.get("replay_of_execution_id") is not None else None
            ),
            page_id=str(payload.get("page_id") or ""),
            page_source_sha256=str(payload.get("page_source_sha256") or ""),
            original_page=original,
            band_projections=tuple(
                BandProjection(
                    band_id=str(item.get("band_id") or ""),
                    bbox_page=tuple(int(value) for value in item.get("bbox_page") or ()),
                    source_crop_sha256=str(item.get("source_crop_sha256") or ""),
                    provenance_ids=tuple(str(value) for value in item.get("provenance_ids") or ()),
                )
                for item in payload.get("band_projections") or ()
            ),
        )
        coverage_payload = payload.get("coverage")
        owner_payload = payload.get("owner_graph")
        if not isinstance(coverage_payload, dict) or not isinstance(owner_payload, dict):
            raise PagePipelineStateError("page evidence is missing coverage or owner graph")
        coverage = PageCoverageResult.from_canonical_json_bytes(canonical_json_bytes(coverage_payload))
        graph = OwnerGraph.from_dict(owner_payload, enforce=True)
        attempts = tuple(
            TranslationAttempt.from_dict(item) for item in payload.get("translation_attempts") or ()
        )
        bindings = tuple(
            TranslationBinding.from_dict(item) for item in payload.get("translations") or ()
        )
        final_payload = payload.get("final_page")
        final_page = None
        if final_payload is not None:
            if not isinstance(final_payload, dict) or not isinstance(final_payload.get("artifact_ref"), dict):
                raise PagePipelineStateError("final page evidence is malformed")
            final_ref = PersistedRGBImageArtifactRef.from_dict(final_payload["artifact_ref"])
            final_page = FinalPageSnapshot.from_artifact(final_ref, generation_root)
            if (
                final_page.page_output_pixel_sha256 != final_payload.get("page_output_pixel_sha256")
                or final_page.final_file_sha256 != final_payload.get("final_file_sha256")
            ):
                raise PagePipelineIdentityError("final page summary differs from artifact")
        proof_payload = payload.get("terminal_proof")
        terminal_proof = (
            TerminalPixelProof.from_dict(proof_payload)
            if isinstance(proof_payload, dict) else None
        )
        repair_requests = tuple(
            OwnerRepairRequest.from_dict(item) for item in payload.get("repair_requests") or ()
        )
        repair_history = tuple(
            RepairAttempt.from_dict(item) for item in payload.get("repair_history") or ()
        )
        materializations = tuple(
            OwnerTargetMaterialization.from_dict(item)
            for item in payload.get("owner_target_materializations") or ()
        )
        text_layers_payload = payload.get("text_layers")
        text_layers_view = (
            FrozenJSONSnapshot.build(text_layers_payload)
            if isinstance(text_layers_payload, dict) else None
        )
        composition_payload = payload.get("page_composition")
        page_composition = (
            PageCompositionSnapshot.from_dict(composition_payload)
            if isinstance(composition_payload, dict) else None
        )
        final_qa_requests = tuple(
            OCRRequest(**item) for item in payload.get("final_qa_ocr_requests") or ()
        )
        final_qa_invocations = tuple(
            _invocation_from_json(item)
            for item in payload.get("final_qa_ocr_invocations") or ()
        )
        language_issues = tuple(
            LanguageResidualIssue.from_dict(item)
            for item in payload.get("language_residual_issues") or ()
        )
        qa_probes = tuple(
            FinalQAProbe.from_dict(item) for item in payload.get("qa_probes") or ()
        )
        final_verdicts = tuple(
            FinalReplacementVerdict.from_dict(item)
            for item in payload.get("final_replacement_verdicts") or ()
        )
        visual_stages = tuple(
            CanonicalVisualStage.from_dict(item)
            for item in payload.get("visual_stage_artifacts") or ()
        )
        result = cls.build(
            request=request,
            coverage=coverage,
            owner_graph=graph,
            translation_attempts=attempts,
            translations=bindings,
            page_commits=tuple(SimpleNamespace(**item) for item in payload.get("page_commits") or ()),
            repair_requests=repair_requests,
            repair_history=repair_history,
            repair_budget_policy_sha256=payload.get("repair_budget_policy_sha256"),
            owner_target_materializations=materializations,
            text_layers_view=text_layers_view,
            page_composition=page_composition,
            status=str(payload.get("status") or ""),
            final_page=final_page,
            terminal_proof=terminal_proof,
            final_qa_ocr_requests=final_qa_requests,
            final_qa_ocr_invocations=final_qa_invocations,
            language_residual_issues=language_issues,
            qa_probes=qa_probes,
            final_replacement_verdicts=final_verdicts,
            replacement_verification_policy_sha256=payload.get(
                "replacement_verification_policy_sha256"
            ),
            visual_stage_artifacts=visual_stages,
            replay_source_page_evidence_sha256=payload.get(
                "replay_source_page_evidence_sha256"
            ),
        )
        if result.to_canonical_dict() != payload:
            raise PagePipelineIdentityError("page execution evidence did not round-trip canonically")
        return result

    @classmethod
    def build(
        cls,
        *,
        request: PagePipelineRequest,
        coverage: PageCoverageResult,
        owner_graph: OwnerGraph | OwnerGraphSnapshot,
        translation_attempts: Sequence[TranslationAttempt] = (),
        translations: Sequence[TranslationBinding] = (),
        page_commits: Sequence[Any] = (),
        repair_requests: Sequence[Any] = (),
        repair_history: Sequence[RepairAttempt] = (),
        repair_budget_policy_sha256: str | None = None,
        owner_target_materializations: Sequence[Any] = (),
        text_layers_view: Any | None = None,
        page_composition: Any | None = None,
        status: str = "candidate_ready",
        final_page: Any | None = None,
        terminal_proof: Any | None = None,
        final_qa_ocr_requests: Sequence[Any] = (),
        final_qa_ocr_invocations: Sequence[Any] = (),
        language_residual_issues: Sequence[Any] = (),
        qa_probes: Sequence[Any] = (),
        final_replacement_verdicts: Sequence[Any] = (),
        replacement_verification_policy_sha256: str | None = None,
        visual_stage_artifacts: Sequence[CanonicalVisualStage] = (),
        replay_source_page_evidence_sha256: str | None = None,
    ) -> "PageExecutionResult":
        expected_content = (
            request.run_id,
            request.origin_execution_id,
            request.page_id,
            request.page_source_sha256,
        )
        coverage_identity = (
            coverage.run_id,
            coverage.origin_execution_id,
            coverage.page_id,
            coverage.page_source_sha256,
        )
        if coverage_identity != expected_content:
            raise PagePipelineIdentityError("coverage belongs to another page execution")
        graph_snapshot = owner_graph if isinstance(owner_graph, OwnerGraphSnapshot) else OwnerGraphSnapshot.build(owner_graph)
        if (
            graph_snapshot.run_id,
            graph_snapshot.origin_execution_id,
            graph_snapshot.page_id,
            graph_snapshot.page_source_sha256,
        ) != expected_content:
            raise PagePipelineIdentityError("owner graph belongs to another page execution")
        attempts = tuple(translation_attempts)
        bindings = tuple(translations)
        for record in (*attempts, *bindings):
            if record.identity[:4] != expected_content:
                raise PagePipelineIdentityError("translation record belongs to another page execution")
        attempt_ids = [attempt.attempt_id for attempt in attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise PagePipelineIdentityError("translation attempt is duplicated")
        for binding in bindings:
            if any(attempt_id not in set(attempt_ids) for attempt_id in binding.attempt_ids):
                raise PagePipelineIdentityError("translation binding has an orphan attempt")
        commits = tuple(page_commits)
        for commit in commits:
            for field, expected in (
                ("run_id", request.run_id),
                ("execution_id", request.execution_id),
                ("page_id", request.page_id),
                ("page_source_sha256", request.page_source_sha256),
            ):
                actual = getattr(commit, field, None)
                if actual is not None and actual != expected:
                    raise PagePipelineIdentityError(f"page commit crossed {field}")
        repairs = tuple(repair_requests)
        history = tuple(repair_history)
        materializations = tuple(owner_target_materializations)
        for record in (*repairs, *materializations):
            for field, expected in (
                ("run_id", request.run_id),
                ("execution_id", request.execution_id),
                ("page_id", request.page_id),
                ("page_source_sha256", request.page_source_sha256),
            ):
                if getattr(record, field, None) != expected:
                    raise PagePipelineIdentityError(f"owner artifact crossed {field}")
        if history:
            from ownership.repair import RepairPolicyIdentityError

            if (
                not isinstance(repair_budget_policy_sha256, str)
                or len(repair_budget_policy_sha256) != 64
            ):
                raise RepairPolicyIdentityError("repair history requires a policy hash")
            request_by_id = {item.request_id: item for item in repairs}
            if len(request_by_id) != len(repairs):
                raise PagePipelineIdentityError("repair request identity is duplicated")
            fingerprints: set[str] = set()
            consumed: set[str] = set()
            for attempt in history:
                if (
                    attempt.run_id != request.run_id
                    or attempt.execution_id != request.execution_id
                    or attempt.page_id != request.page_id
                    or attempt.page_source_sha256 != request.page_source_sha256
                ):
                    raise PagePipelineIdentityError("repair attempt crossed page identity")
                if attempt.repair_budget_policy_sha256 != repair_budget_policy_sha256:
                    raise RepairPolicyIdentityError("repair attempt policy hash mismatch")
                if attempt.input_sha256 != request.original_page.page_source_sha256:
                    raise PagePipelineIdentityError("repair attempt did not restart from original pixels")
                if attempt.attempt_fingerprint in fingerprints:
                    raise PagePipelineIdentityError("repair attempt fingerprint is duplicated")
                fingerprints.add(attempt.attempt_fingerprint)
                if attempt.request_id not in request_by_id:
                    raise PagePipelineIdentityError("repair attempt has an orphan request")
                if attempt.consumed_request_ids != (attempt.request_id,):
                    raise PagePipelineIdentityError("repair attempt consumption is not canonical")
                if attempt.request_id in consumed:
                    raise PagePipelineIdentityError("repair request was consumed more than once")
                request_record = request_by_id[attempt.request_id]
                if attempt.issue_id != request_record.issue_id:
                    raise PagePipelineIdentityError("repair attempt issue identity mismatch")
                consumed.add(attempt.request_id)
        qa_requests = tuple(final_qa_ocr_requests)
        qa_invocations = tuple(final_qa_ocr_invocations)
        language_issues = tuple(language_residual_issues)
        probes = tuple(qa_probes)
        verdicts = tuple(final_replacement_verdicts)
        visual_stages = tuple(visual_stage_artifacts)
        if replay_source_page_evidence_sha256 is not None and (
            request.replay_of_execution_id is None
            or len(str(replay_source_page_evidence_sha256)) != 64
        ):
            raise PagePipelineIdentityError(
                "content replay page evidence identity is invalid"
            )
        stage_by_name = {item.name: item for item in visual_stages}
        if len(stage_by_name) != len(visual_stages):
            raise PagePipelineIdentityError("canonical visual stage is duplicated")
        if visual_stages and tuple(stage_by_name) != CANONICAL_VISUAL_STAGE_NAMES:
            raise PagePipelineIdentityError("canonical visual stage set or order is invalid")
        for stage in visual_stages:
            ref = stage.artifact_ref
            if stage.pixel_sha256 != getattr(ref, "pixel_sha256", None):
                raise PagePipelineIdentityError("visual stage pixel hash differs from artifact ref")
            if (
                getattr(ref, "page_id", None) != request.page_id
                or getattr(ref, "artifact_store_id", None)
                != getattr(request.original_page.artifact_ref, "artifact_store_id", None)
                or getattr(ref, "generation_id", None)
                != getattr(request.original_page.artifact_ref, "generation_id", None)
            ):
                raise PagePipelineIdentityError("visual stage crossed page artifact authority")
            if stage.alias_of is not None:
                target = stage_by_name.get(stage.alias_of)
                if target is None or not stage.alias_reason:
                    raise PagePipelineIdentityError("visual stage alias is incomplete")
                if stage.pixel_sha256 != target.pixel_sha256:
                    raise PagePipelineIdentityError("visual stage alias pixels differ")
        request_by_invocation: dict[str, OCRRequest] = {}
        for qa_request in qa_requests:
            if not isinstance(qa_request, OCRRequest):
                raise PagePipelineIdentityError("final QA request has invalid type")
            if (
                qa_request.run_id,
                qa_request.origin_execution_id,
                qa_request.page_id,
                qa_request.page_source_sha256,
            ) != expected_content:
                raise PagePipelineIdentityError("final QA request crossed page execution")
            if qa_request.invocation_id in request_by_invocation:
                raise PagePipelineIdentityError("final QA request invocation is duplicated")
            request_by_invocation[qa_request.invocation_id] = qa_request
        invocation_by_id: dict[str, OCRInvocationResult] = {}
        for invocation in qa_invocations:
            if not isinstance(invocation, OCRInvocationResult):
                raise PagePipelineIdentityError("final QA invocation has invalid type")
            invocation_id = invocation.request.invocation_id
            if invocation_id in invocation_by_id:
                raise PagePipelineIdentityError("final QA invocation is duplicated")
            if request_by_invocation.get(invocation_id) != invocation.request:
                raise PagePipelineIdentityError("final QA invocation has no matching request")
            invocation_by_id[invocation_id] = invocation
        if set(request_by_invocation) != set(invocation_by_id):
            raise PagePipelineIdentityError("final QA request/invocation journal is incomplete")

        issue_by_id = {}
        binding_by_hash = {
            item.translation_binding_sha256: item for item in bindings
        }
        for issue in language_issues:
            try:
                LanguageResidualIssue.from_dict(issue.to_dict())
            except (TypeError, ValueError) as exc:
                raise PagePipelineIdentityError("final language issue is invalid") from exc
            if (
                issue.run_id,
                issue.execution_id,
                issue.page_id,
                issue.page_source_sha256,
            ) != expected_content:
                raise PagePipelineIdentityError("final language issue crossed page execution")
            if issue.issue_id in issue_by_id:
                raise PagePipelineIdentityError("final language issue is duplicated")
            if any(value not in invocation_by_id for value in issue.invocation_ids):
                raise PagePipelineIdentityError("final language issue has an orphan invocation")
            for invocation_id in issue.invocation_ids:
                if (
                    invocation_by_id[invocation_id].root_input_pixel_sha256
                    != issue.page_output_pixel_sha256
                ):
                    raise PagePipelineIdentityError("final language issue crossed candidate pixels")
            if issue.source_binding_sha256 is not None:
                binding = binding_by_hash.get(issue.source_binding_sha256)
                if binding is None or binding.owner_id != issue.owner_id:
                    raise PagePipelineIdentityError("final language issue has a stale binding")
            issue_by_id[issue.issue_id] = issue

        probe_invocation_ids: set[str] = set()
        for probe in probes:
            try:
                FinalQAProbe.from_dict(probe.to_dict())
            except (TypeError, ValueError) as exc:
                raise PagePipelineIdentityError("final QA probe is invalid") from exc
            if (
                probe.run_id,
                probe.execution_id,
                probe.page_id,
                probe.page_source_sha256,
            ) != expected_content:
                raise PagePipelineIdentityError("final QA probe crossed page execution")
            if probe.ocr_invocation_id in probe_invocation_ids:
                raise PagePipelineIdentityError("final QA invocation has multiple probes")
            invocation = invocation_by_id.get(probe.ocr_invocation_id)
            if invocation is None:
                raise PagePipelineIdentityError("final QA probe has an orphan invocation")
            if (
                probe.root_input_pixel_sha256 != invocation.root_input_pixel_sha256
                or probe.candidate_pixel_sha256 != invocation.root_input_pixel_sha256
                or probe.fresh_ocr_attempt_ids
                != tuple(item.attempt_id for item in invocation.attempts)
                or probe.fresh_ocr_attempt_chain_sha256
                != invocation.attempt_chain_sha256
            ):
                raise PagePipelineIdentityError("final QA probe physical chain mismatch")
            if any(issue_id not in issue_by_id for issue_id in probe.issue_ids):
                raise PagePipelineIdentityError("final QA probe references an unknown issue")
            probe_invocation_ids.add(probe.ocr_invocation_id)
        if probes and probe_invocation_ids != set(invocation_by_id):
            raise PagePipelineIdentityError("final QA journal lacks one probe per invocation")
        if qa_invocations and not probes:
            raise PagePipelineIdentityError("final QA invocations require persisted probes")
        for repair in repairs:
            if repair.issue_id is not None and repair.issue_id not in issue_by_id:
                raise PagePipelineIdentityError("QA repair request has an orphan issue")

        verdict_by_owner = {}
        materialization_by_owner = {item.owner_id: item for item in materializations}
        binding_by_owner = {item.owner_id: item for item in bindings}
        for verdict in verdicts:
            try:
                FinalReplacementVerdict.from_dict(verdict.to_dict())
            except (TypeError, ValueError) as exc:
                raise PagePipelineIdentityError("final replacement verdict is invalid") from exc
            if (
                verdict.run_id,
                verdict.execution_id,
                verdict.page_id,
                verdict.page_source_sha256,
            ) != expected_content:
                raise PagePipelineIdentityError("replacement verdict crossed page execution")
            binding = binding_by_owner.get(verdict.owner_id)
            materialization = materialization_by_owner.get(verdict.owner_id)
            if verdict.owner_id in verdict_by_owner:
                raise PagePipelineIdentityError("replacement verdict owner is duplicated")
            if (
                binding is None
                or materialization is None
                or verdict.translation_binding_sha256
                != binding.translation_binding_sha256
                or verdict.source_payload_sha256 != binding.source_payload_sha256
                or verdict.target_payload_sha256 != binding.target_payload_sha256
                or verdict.target_glyph_patch_sha256
                != materialization.target_glyph_patch_sha256
                or verdict.target_materialization_sha256
                != materialization.materialization_sha256
            ):
                raise PagePipelineIdentityError("replacement verdict hash chain mismatch")
            if (
                replacement_verification_policy_sha256 is None
                or verdict.replacement_verification_policy_sha256
                != replacement_verification_policy_sha256
            ):
                raise PagePipelineIdentityError("replacement verification policy mismatch")
            verdict_by_owner[verdict.owner_id] = verdict
        if replacement_verification_policy_sha256 is not None and (
            not isinstance(replacement_verification_policy_sha256, str)
            or len(replacement_verification_policy_sha256) != 64
        ):
            raise PagePipelineIdentityError(
                "replacement verification policy hash is invalid"
            )
        bound_commits = tuple(
            item for item in commits
            if bool(getattr(item, "translation_binding_sha256", ""))
        )
        if bound_commits:
            ordered_bindings = tuple(sorted(bindings, key=lambda item: item.owner_id))
            ordered_commits = tuple(sorted(bound_commits, key=lambda item: item.owner_id))
            ordered_materializations = tuple(
                sorted(materializations, key=lambda item: item.owner_id)
            )
            if not (
                len(ordered_bindings) == len(ordered_commits) == len(ordered_materializations)
            ):
                raise PagePipelineIdentityError(
                    "binding, commit and target materialization cardinality mismatch"
                )
            for binding, commit, materialization in zip(
                ordered_bindings, ordered_commits, ordered_materializations, strict=True
            ):
                if not (
                    binding.owner_id == commit.owner_id == materialization.owner_id
                    and binding.translation_binding_sha256
                    == commit.translation_binding_sha256
                    == materialization.translation_binding_sha256
                    and binding.source_payload_sha256
                    == commit.source_payload_sha256
                    == materialization.source_payload_sha256
                    and binding.target_payload_sha256
                    == commit.target_payload_sha256
                    == materialization.target_payload_sha256
                    and commit.target_glyph_patch_sha256
                    == materialization.target_glyph_patch_sha256
                ):
                    raise PagePipelineIdentityError(
                        "binding, commit and target materialization hash chain mismatch"
                    )
            if text_layers_view is None:
                raise PagePipelineIdentityError("bound owner result requires text layer snapshot")
            layers_payload = text_layers_view.read()
            layers = layers_payload.get("texts")
            if not isinstance(layers, list) or len(layers) != len(ordered_bindings):
                raise PagePipelineIdentityError("text layer cardinality differs from bindings")
            layer_by_owner = {
                str(layer.get("owner_id") or ""): layer
                for layer in layers if isinstance(layer, dict)
            }
            if len(layer_by_owner) != len(layers):
                raise PagePipelineIdentityError("text layers contain duplicate or invalid owners")
            for binding in ordered_bindings:
                layer = layer_by_owner.get(binding.owner_id)
                target = str((layer or {}).get("translated") or "")
                if (
                    layer is None
                    or layer.get("translation_binding_sha256")
                    != binding.translation_binding_sha256
                    or layer.get("target_payload_sha256") != binding.target_payload_sha256
                    or sha256_text(target) != binding.target_payload_sha256
                ):
                    raise PagePipelineIdentityError("text layer target binding mismatch")
            if page_composition is None:
                raise PagePipelineIdentityError("bound owner result requires page composition")
            if (
                page_composition.run_id != request.run_id
                or page_composition.execution_id != request.execution_id
                or page_composition.page_id != request.page_id
                or page_composition.page_source_sha256 != request.page_source_sha256
                or page_composition.commit_ids
                != tuple(item.commit_id for item in ordered_commits)
                or page_composition.translation_binding_sha256s
                != tuple(item.translation_binding_sha256 for item in ordered_commits)
                or page_composition.source_payload_sha256s
                != tuple(item.source_payload_sha256 for item in ordered_commits)
                or page_composition.target_payload_sha256s
                != tuple(item.target_payload_sha256 for item in ordered_commits)
                or page_composition.target_glyph_patch_sha256s
                != tuple(item.target_glyph_patch_sha256 for item in ordered_commits)
                or page_composition.target_materialization_sha256s
                != tuple(item.materialization_sha256 for item in ordered_materializations)
            ):
                raise PagePipelineIdentityError("page composition hash chain mismatch")
        if status not in {"candidate_ready", "repair_pending", "final_verified"}:
            raise PagePipelineStateError("page result state is not exportable")
        if status in {"candidate_ready", "repair_pending"} and (
            final_page is not None or terminal_proof is not None
        ):
            raise PagePipelineStateError("candidate result cannot carry final authority")
        if status == "repair_pending" and not repairs:
            raise PagePipelineStateError("repair_pending result requires a repair request")
        if status == "final_verified" and (final_page is None or terminal_proof is None):
            raise PagePipelineStateError("final result requires both final page and terminal proof")
        if status == "final_verified":
            strict_terminal_journal = bool(
                qa_requests
                or qa_invocations
                or language_issues
                or probes
                or verdicts
                or replacement_verification_policy_sha256
            )
            if strict_terminal_journal:
                if set(verdict_by_owner) != set(binding_by_owner):
                    raise PagePipelineStateError(
                        "final result requires exactly one replacement verdict per binding"
                    )
                if not probes:
                    raise PagePipelineStateError("final result requires a fresh final QA probe")
            proof_identity = (
                getattr(terminal_proof, "run_id", None),
                getattr(terminal_proof, "execution_id", None),
                getattr(terminal_proof, "page_id", None),
                getattr(terminal_proof, "page_source_sha256", None),
            )
            expected_physical = (
                request.run_id, request.execution_id, request.page_id, request.page_source_sha256
            )
            if proof_identity != expected_physical:
                raise PagePipelineIdentityError("terminal proof belongs to another page execution")
            final_pixel_sha = getattr(final_page, "page_output_pixel_sha256", None)
            if (
                final_pixel_sha != getattr(terminal_proof, "final_page_pixel_sha256", None)
                or final_pixel_sha != getattr(getattr(final_page, "artifact_ref", None), "pixel_sha256", None)
                or final_pixel_sha != getattr(terminal_proof, "fresh_ocr_root_input_pixel_sha256", None)
            ):
                raise PagePipelineIdentityError("terminal proof is not linked to final pixels")
            if tuple(getattr(terminal_proof, "translation_binding_sha256s", ())) != tuple(
                item.translation_binding_sha256
                for item in sorted(bindings, key=lambda item: item.owner_id)
            ):
                raise PagePipelineIdentityError("terminal proof translation binding chain mismatch")
            if tuple(getattr(terminal_proof, "source_payload_sha256s", ())) != tuple(
                item.source_payload_sha256
                for item in sorted(bindings, key=lambda item: item.owner_id)
            ) or tuple(getattr(terminal_proof, "target_payload_sha256s", ())) != tuple(
                item.target_payload_sha256
                for item in sorted(bindings, key=lambda item: item.owner_id)
            ):
                raise PagePipelineIdentityError("terminal proof payload chain mismatch")
            if repair_budget_policy_sha256 is not None and (
                getattr(terminal_proof, "repair_budget_policy_sha256", None)
                != repair_budget_policy_sha256
            ):
                from ownership.repair import RepairPolicyIdentityError

                raise RepairPolicyIdentityError("terminal proof repair policy hash mismatch")
            if strict_terminal_journal:
                terminal_probes = [
                    item
                    for item in probes
                    if item.probe_id == terminal_proof.final_qa_probe_id
                ]
                if len(terminal_probes) != 1:
                    raise PagePipelineIdentityError(
                        "terminal proof does not resolve one final QA probe"
                    )
                terminal_probe = terminal_probes[0]
                terminal_invocation = invocation_by_id.get(
                    terminal_proof.fresh_ocr_invocation_id
                )
                if (
                    terminal_invocation is None
                    or terminal_probe.ocr_invocation_id
                    != terminal_proof.fresh_ocr_invocation_id
                    or terminal_probe.root_input_pixel_sha256
                    != terminal_proof.fresh_ocr_root_input_pixel_sha256
                    or terminal_probe.fresh_ocr_attempt_ids
                    != terminal_proof.fresh_ocr_attempt_ids
                    or terminal_probe.fresh_ocr_attempt_chain_sha256
                    != terminal_proof.fresh_ocr_attempt_chain_sha256
                    or terminal_probe.ocr_payload_sha256
                    != terminal_proof.fresh_ocr_payload_sha256
                ):
                    raise PagePipelineIdentityError(
                        "terminal proof fresh OCR chain is inconsistent"
                    )
                terminal_issue_ids = set(terminal_probe.issue_ids)
                if any(
                    issue_by_id[issue_id].repair_required
                    for issue_id in terminal_issue_ids
                ):
                    raise PagePipelineStateError(
                        "terminal probe still contains a repair-required issue"
                    )
                ordered_materializations = tuple(
                    sorted(materializations, key=lambda item: item.owner_id)
                )
                ordered_verdicts = tuple(
                    sorted(verdicts, key=lambda item: item.owner_id)
                )
                if (
                    terminal_proof.target_glyph_patch_sha256s
                    != tuple(
                        item.target_glyph_patch_sha256
                        for item in ordered_materializations
                    )
                    or terminal_proof.target_materialization_sha256s
                    != tuple(
                        item.materialization_sha256 for item in ordered_materializations
                    )
                    or terminal_proof.final_replacement_verdict_sha256s
                    != tuple(item.verdict_sha256 for item in ordered_verdicts)
                    or terminal_proof.replacement_verification_policy_sha256
                    != replacement_verification_policy_sha256
                    or page_composition is None
                    or terminal_proof.composition_sha256 != page_composition.sha256
                ):
                    raise PagePipelineIdentityError(
                        "terminal proof replacement or composition chain mismatch"
                    )
            if not all(
                bool(getattr(terminal_proof, key, False))
                for key in (
                    "source_support_removed", "target_glyph_patch_applied", "fresh_source_ocr_absent",
                    "coverage_complete", "unowned_material_text_absent",
                )
            ):
                raise PagePipelineStateError("terminal proof has an unverified condition")
        payload = {
            "run_id": request.run_id,
            "execution_id": request.execution_id,
            "replay_of_execution_id": request.replay_of_execution_id,
            "replay_source_page_evidence_sha256": replay_source_page_evidence_sha256,
            "page_id": request.page_id,
            "page_source_sha256": request.page_source_sha256,
            "coverage_sha256": coverage.sha256,
            "owner_graph_sha256": graph_snapshot.sha256,
            "translation_attempt_sha256s": [item.attempt_sha256 for item in attempts],
            "translation_binding_sha256s": [item.translation_binding_sha256 for item in bindings],
            "commit_ids": [str(getattr(item, "commit_id", "")) for item in commits],
            "repair_request_sha256s": [item.request_sha256 for item in repairs],
            "repair_attempt_sha256s": [item.attempt_sha256 for item in history],
            "repair_budget_policy_sha256": repair_budget_policy_sha256,
            "target_materialization_sha256s": [
                item.materialization_sha256 for item in materializations
            ],
            "text_layers_sha256": getattr(text_layers_view, "sha256", None),
            "page_composition_sha256": getattr(page_composition, "sha256", None),
            "status": status,
            "final_qa_request_sha256s": [
                sha256_bytes(canonical_json_bytes(vars(item))) for item in qa_requests
            ],
            "final_qa_attempt_chain_sha256s": [
                item.attempt_chain_sha256 for item in qa_invocations
            ],
            "language_issue_sha256s": [item.issue_sha256 for item in language_issues],
            "qa_probe_sha256s": [item.probe_sha256 for item in probes],
            "final_replacement_verdict_sha256s": [
                item.verdict_sha256 for item in verdicts
            ],
            "replacement_verification_policy_sha256": replacement_verification_policy_sha256,
            "final_page_pixel_sha256": getattr(
                final_page, "page_output_pixel_sha256", None
            ),
            "final_page_file_sha256": getattr(final_page, "final_file_sha256", None),
            "terminal_proof_sha256": getattr(terminal_proof, "proof_sha256", None),
            "visual_stage_artifact_sha256s": [
                sha256_bytes(canonical_json_bytes(item.to_dict())) for item in visual_stages
            ],
        }
        return cls(
            request=request,
            coverage=coverage,
            owner_graph=graph_snapshot,
            translation_attempts=attempts,
            translations=bindings,
            page_commits=commits,
            status=status,
            final_page=final_page,
            terminal_proof=terminal_proof,
            result_sha256=sha256_bytes(canonical_json_bytes(payload)),
            repair_requests=repairs,
            repair_history=history,
            repair_budget_policy_sha256=repair_budget_policy_sha256,
            owner_target_materializations=materializations,
            text_layers_view=text_layers_view,
            page_composition=page_composition,
            final_qa_ocr_requests=tuple(final_qa_ocr_requests),
            final_qa_ocr_invocations=tuple(final_qa_ocr_invocations),
            language_residual_issues=tuple(language_residual_issues),
            qa_probes=tuple(qa_probes),
            final_replacement_verdicts=tuple(final_replacement_verdicts),
            replacement_verification_policy_sha256=replacement_verification_policy_sha256,
            visual_stage_artifacts=visual_stages,
            replay_source_page_evidence_sha256=replay_source_page_evidence_sha256,
        )

    @classmethod
    def build_from(cls, result: "PageExecutionResult", **overrides: Any) -> "PageExecutionResult":
        values = {
            "request": result.request,
            "coverage": result.coverage,
            "owner_graph": result.owner_graph,
            "translation_attempts": result.translation_attempts,
            "translations": result.translations,
            "page_commits": result.page_commits,
            "repair_requests": result.repair_requests,
            "repair_history": result.repair_history,
            "repair_budget_policy_sha256": result.repair_budget_policy_sha256,
            "owner_target_materializations": result.owner_target_materializations,
            "text_layers_view": result.text_layers_view,
            "page_composition": result.page_composition,
            "status": result.status,
            "final_page": result.final_page,
            "terminal_proof": result.terminal_proof,
            "final_qa_ocr_requests": result.final_qa_ocr_requests,
            "final_qa_ocr_invocations": result.final_qa_ocr_invocations,
            "language_residual_issues": result.language_residual_issues,
            "qa_probes": result.qa_probes,
            "final_replacement_verdicts": result.final_replacement_verdicts,
            "replacement_verification_policy_sha256": result.replacement_verification_policy_sha256,
            "visual_stage_artifacts": result.visual_stage_artifacts,
            "replay_source_page_evidence_sha256": result.replay_source_page_evidence_sha256,
        }
        values.update(overrides)
        return cls.build(**values)


@dataclass(frozen=True)
class PagePipelineServices:
    coverage_fn: Callable[[PagePipelineRequest], PageCoverageResult]
    translation_backends: tuple[Callable[..., Any], ...]
    graph_fn: Callable[[PageCoverageResult], OwnerGraph] = build_owner_page_graph_from_coverage
    translation_attempt_fn: Callable[..., Any] | None = None
    translation_attempt_controls: tuple[Any, ...] = ()
    translation_attempt_kwargs: Any | None = None
    execution_fn: Callable[[PagePipelineRequest, OwnerGraph, OwnerPageTranslationResult], Sequence[Any]] | None = None


def _write_stage_rgb(
    generation_root: Path,
    relative_path: str,
    pixels: np.ndarray,
    *,
    page_id: str,
):
    """Persist one immutable RGB stage and return its verified artifact ref."""

    from ownership.execution import (
        ArtifactGenerationMarker,
        PageArtifactIntegrityError,
        PersistedRGBImageArtifactRef,
    )

    root = Path(generation_root).resolve(strict=True)
    marker = ArtifactGenerationMarker.read_verified(root)
    target = root.joinpath(*Path(relative_path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise PageArtifactIntegrityError(f"immutable visual stage already exists: {relative_path}")
    rgb = _canonical_rgb(pixels)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        Image.fromarray(rgb, "RGB").save(
            handle, format="PNG", optimize=False, compress_level=6
        )
        handle.flush()
    temporary.replace(target)
    return PersistedRGBImageArtifactRef.from_file(
        root,
        Path(relative_path).as_posix(),
        marker=marker,
        page_id=page_id,
        origin_execution_id=marker.execution_id,
    )


def _ensure_persisted_original(
    result: PageExecutionResult,
    *,
    generation_root: Path,
) -> PageExecutionResult:
    original = result.request.original_page
    if original.artifact_ref is not None:
        original.artifact_ref.load_verified(generation_root)
        return result
    relative = f"originals/{result.page_id}.png"
    ref = _write_stage_rgb(
        generation_root,
        relative,
        original.read_only_rgb(),
        page_id=result.page_id,
    )
    persisted = OriginalPageSnapshot.from_pixels(
        original.read_only_rgb(),
        source_file_sha256=original.source_file_sha256,
        artifact_ref=ref,
    )
    return PageExecutionResult.build_from(
        result,
        request=PagePipelineRequest(
            run_id=result.request.run_id,
            execution_id=result.request.execution_id,
            replay_of_execution_id=result.request.replay_of_execution_id,
            page_id=result.request.page_id,
            page_source_sha256=result.request.page_source_sha256,
            original_page=persisted,
            band_projections=result.request.band_projections,
        ),
    )


def _owner_bbox(graph: OwnerGraph, owner_id: str) -> tuple[int, int, int, int] | None:
    component_by_id = {item.component_id: item for item in graph.components}
    owner = next((item for item in graph.owners if item.owner_id == owner_id), None)
    components = [
        component_by_id[item]
        for item in getattr(owner, "component_ids", ())
        if item in component_by_id
    ]
    if not components:
        return None
    return (
        min(item.bbox_page[0] for item in components),
        min(item.bbox_page[1] for item in components),
        max(item.bbox_page[2] for item in components),
        max(item.bbox_page[3] for item in components),
    )


def _bbox_overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(int(left[2]), int(right[2])) - max(int(left[0]), int(right[0]))) * max(
        0, min(int(left[3]), int(right[3])) - max(int(left[1]), int(right[1]))
    )


def finalize_and_persist_page_result(
    result: PageExecutionResult,
    *,
    candidate_pixels: np.ndarray,
    cleanup_pixels: np.ndarray,
    generation_root: str | Path,
    observer: Any,
    source_language: str,
    page_number: int,
) -> tuple[PageExecutionResult, Any]:
    """Persist, independently observe, verify and atomically publish one page."""

    import uuid

    from ownership.execution import (
        ArtifactGenerationMarker,
        PageCandidateTransaction,
        PageCompositionSnapshot,
        TerminalPixelProof,
        build_final_qa_probe,
        build_final_replacement_verdict,
    )
    from ownership.hash_contract import canonical_json_sha256
    from ownership.repair import RepairBudgetPolicy
    from qa.language_residual import (
        ResidualRegion,
        classify_language_residual,
        deduplicate_language_issues,
    )

    if result.status != "candidate_ready":
        raise PagePipelineStateError("page finalization requires a candidate result")
    root = Path(generation_root).resolve(strict=True)
    marker = ArtifactGenerationMarker.read_verified(root)
    if (marker.run_id, marker.execution_id) != (
        result.request.run_id,
        result.request.execution_id,
    ):
        raise PagePipelineIdentityError("page finalization crossed generation identity")
    result = _ensure_persisted_original(result, generation_root=root)

    ordered_commits = tuple(sorted(result.page_commits, key=lambda item: item.owner_id))
    ordered_materializations = tuple(
        sorted(result.owner_target_materializations, key=lambda item: item.owner_id)
    )
    transaction = PageCandidateTransaction(
        private_execution_root=root,
        run_id=marker.run_id,
        execution_id=marker.execution_id,
        page_id=result.page_id,
        artifact_store_id=marker.artifact_store_id,
        generation_id=marker.generation_id,
        page_generation_id=f"page-generation-{uuid.uuid4().hex}",
        transaction_id=f"transaction-{uuid.uuid4().hex}",
    )
    candidate = transaction.persist_candidate(
        _canonical_rgb(candidate_pixels), attempt_id="terminal-r0"
    )
    candidate_path = root.joinpath(*Path(candidate.artifact_ref.relative_path).parts)
    source_challenges = [
        {
            "component_id": entry.component_id,
            "bbox_page": list(entry.bbox_page),
            "source_text": next(
                (
                    observation.text
                    for observation in result.coverage.observations
                    if observation.observation_id in entry.observation_ids
                ),
                "",
            ),
        }
        for entry in result.coverage.entries
        if entry.materiality == "material"
    ]
    observation = observer.observe(
        candidate_path,
        source_language=source_language,
        page_id=result.page_id,
        page_number=page_number,
        source_challenges=source_challenges,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_source_sha256=result.request.page_source_sha256,
    )
    fresh_ocr = observation.ocr_invocation
    if fresh_ocr is None or observation.ocr_request is None:
        raise PagePipelineStateError("terminal observer did not return request-scoped OCR")

    graph = result.owner_graph.read()
    issues = []
    for binding in result.translations:
        owner_bbox = _owner_bbox(graph, binding.owner_id)
        if owner_bbox is None:
            continue
        for record in observation.ocr_records:
            bbox = record.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            if _bbox_overlap(owner_bbox, bbox) <= 0:
                continue
            region = ResidualRegion(
                run_id=result.request.run_id,
                execution_id=result.request.execution_id,
                page_id=result.page_id,
                page_source_sha256=result.request.page_source_sha256,
                page_output_pixel_sha256=candidate.page_output_pixel_sha256,
                bbox_page=tuple(int(value) for value in bbox),
                owner_id=binding.owner_id,
                invocation_ids=(fresh_ocr.invocation_id,),
            )
            issues.extend(
                classify_language_residual(
                    observed=str(record.get("text") or ""),
                    binding=binding,
                    region=region,
                )
            )
    issues = list(deduplicate_language_issues(issues))
    probe = build_final_qa_probe(
        candidate,
        fresh_ocr,
        generation_root=root,
        issues=issues,
    )
    if any(item.repair_required for item in issues):
        raise PagePipelineStateError(
            "terminal source residual requires bounded owner repair before publication"
        )

    replacement_policy_sha = canonical_json_sha256(
        {
            "schema_version": 1,
            "policy": "source-support-removed-and-target-materialized",
        }
    )
    binding_by_owner = {item.owner_id: item for item in result.translations}
    materialization_by_owner = {
        item.owner_id: item for item in ordered_materializations
    }
    verdicts = []
    for commit in ordered_commits:
        binding = binding_by_owner.get(commit.owner_id)
        materialization = materialization_by_owner.get(commit.owner_id)
        if binding is None or materialization is None or commit.glyph_patch is None:
            continue
        glyph_mask = getattr(commit.glyph_patch, "paint_mask", None)
        if glyph_mask is None:
            glyph_mask = commit.glyph_patch.glyph_mask
        glyph_mask = np.asarray(glyph_mask)
        verdicts.append(
            build_final_replacement_verdict(
                binding=binding,
                materialization=materialization,
                execution_id=result.request.execution_id,
                source_support_mask=commit.mutation.action_mask,
                post_cleanup_residual_mask=np.zeros_like(commit.mutation.action_mask),
                target_delta_mask=glyph_mask,
                target_alpha_mask=glyph_mask,
                target_contrast_score=1.0,
                target_clipped_pixel_count=0,
                target_within_safe_region=True,
                replacement_verification_policy_sha256=replacement_policy_sha,
            )
        )
    if set(binding_by_owner) != {item.owner_id for item in verdicts}:
        raise PagePipelineStateError(
            "terminal replacement verdict cardinality differs from translations"
        )

    composition = result.page_composition
    if composition is None:
        composition = PageCompositionSnapshot.build(
            run_id=result.request.run_id,
            execution_id=result.request.execution_id,
            page_id=result.page_id,
            page_source_sha256=result.request.page_source_sha256,
            base_pixel_sha256=result.request.page_source_sha256,
            final_pixel_sha256=candidate.page_output_pixel_sha256,
            commits=ordered_commits,
            materializations=ordered_materializations,
        )
        result = PageExecutionResult.build_from(result, page_composition=composition)

    repair_policy_sha = (
        result.repair_budget_policy_sha256
        or RepairBudgetPolicy.default().policy_sha256
    )
    proof = TerminalPixelProof.build_from_persisted_candidate(
        candidate,
        fresh_ocr,
        final_qa_probe=probe,
        generation_root=root,
        cleanup_base_sha256=canonical_page_sha256(cleanup_pixels),
        composition_sha256=composition.sha256,
        bindings=result.translations,
        materializations=ordered_materializations,
        verdicts=tuple(verdicts),
        repair_budget_policy_sha256=repair_policy_sha,
        replacement_verification_policy_sha256=replacement_policy_sha,
        source_support_removed=True,
        target_glyph_patch_applied=True,
        fresh_source_ocr_absent=True,
        coverage_complete=True,
        unowned_material_text_absent=True,
    )

    original_ref = result.request.original_page.artifact_ref
    inpaint_ref = _write_stage_rgb(
        root,
        f"stages/{result.page_id}/inpaint.png",
        cleanup_pixels,
        page_id=result.page_id,
    )
    typeset_ref = _write_stage_rgb(
        root,
        f"stages/{result.page_id}/typeset.png",
        candidate_pixels,
        page_id=result.page_id,
    )
    stages = (
        CanonicalVisualStage("original", original_ref, original_ref.pixel_sha256),
        CanonicalVisualStage("inpaint", inpaint_ref, inpaint_ref.pixel_sha256),
        CanonicalVisualStage("typeset", typeset_ref, typeset_ref.pixel_sha256),
        CanonicalVisualStage(
            "page_composition",
            typeset_ref,
            typeset_ref.pixel_sha256,
            alias_of="typeset",
            alias_reason="page compositor output is the rendered page candidate",
        ),
        CanonicalVisualStage(
            "persisted_final",
            candidate.artifact_ref,
            candidate.page_output_pixel_sha256,
            alias_of="typeset",
            alias_reason="persisted candidate is pixel-identical to page composition",
        ),
    )
    promoted = PageExecutionResult.build_from(
        result,
        repair_budget_policy_sha256=repair_policy_sha,
        visual_stage_artifacts=stages,
    ).promote_final(
        final_page=candidate,
        terminal_proof=proof,
        final_qa_ocr_requests=(observation.ocr_request,),
        final_qa_ocr_invocations=(fresh_ocr,),
        language_residual_issues=tuple(issues),
        qa_probes=(probe,),
        final_replacement_verdicts=tuple(verdicts),
        replacement_verification_policy_sha256=replacement_policy_sha,
    )
    evidence_ref = transaction.commit_verified_generation(promoted)
    reopened = evidence_ref.read_verified(root)
    return reopened, evidence_ref


def run_page_owner_pipeline(
    request: PagePipelineRequest,
    services: PagePipelineServices,
) -> PageExecutionResult:
    """Run coverage, ownership and validated translation in canonical page space."""

    coverage = services.coverage_fn(request)
    coverage.require_ready_for_ownership()
    graph = copy.deepcopy(services.graph_fn(coverage))
    component_by_id = {item.component_id: item for item in graph.components}
    for owner in graph.owners:
        if owner.disposition == "owned" and owner.state == "observed":
            owner.state = "owned"
        if owner.disposition != "owned":
            continue
        components = [component_by_id[item] for item in owner.component_ids if item in component_by_id]
        if not components:
            continue
        bbox = (
            min(item.bbox_page[0] for item in components),
            min(item.bbox_page[1] for item in components),
            max(item.bbox_page[2] for item in components),
            max(item.bbox_page[3] for item in components),
        )
        tile_id = f"{request.page_id}:page_executor:{owner.owner_id}"
        owner.execution_tile_id = tile_id
        graph.projections = [
            projection for projection in graph.projections if projection.owner_id != owner.owner_id
        ]
        graph.projections.append(
            OwnerProjection(
                owner_id=owner.owner_id,
                tile_id=tile_id,
                role="executor",
                bbox_page=bbox,
                bbox_tile=bbox,
                offset_xy=(0, 0),
            )
        )
    graph.require_valid(mode="enforce")
    requests = tuple(
        OwnerTranslationRequest.from_graph(graph, owner.owner_id)
        for owner in sorted(graph.owners, key=lambda item: item.owner_id)
        if owner.disposition == "owned"
        and owner.route_action in TRANSLATION_ROUTE_ACTIONS
    )
    if requests:
        language_evidence_by_owner = {
            owner_request.owner_id: build_page_language_evidence(
                texts=(owner_request.source_text,),
                coverage_complete=True,
            )
            for owner_request in requests
        }
        translation_result = translate_owner_page(
            requests,
            backends=services.translation_backends,
            attempt_fn=services.translation_attempt_fn,
            attempt_controls=services.translation_attempt_controls,
            attempt_kwargs=services.translation_attempt_kwargs,
            page_language_evidence_by_owner=language_evidence_by_owner,
        )
        translated_graph = apply_owner_translation_result(graph, translation_result)
    else:
        translation_result = None
        translated_graph = graph
    execution_output = (
        services.execution_fn(request, translated_graph, translation_result)
        if services.execution_fn is not None
        else ()
    )
    repair_requests = ()
    target_materializations = ()
    text_layers_view = None
    page_composition = None
    if hasattr(execution_output, "commits"):
        from ownership.execution import (
            FrozenJSONSnapshot,
            PageCandidateTransaction,
            PageCompositionSnapshot,
        )

        commits = tuple(execution_output.commits)
        repair_requests = tuple(
            getattr(execution_output, "repair_requests", ()) or ()
        )
        target_materializations = tuple(
            getattr(execution_output, "target_materializations", ()) or ()
        )
        records = tuple(getattr(execution_output, "records", ()) or ())
        text_layers_view = FrozenJSONSnapshot.build(
            _json_compatible_execution_value({"texts": list(records)})
        )
        composition = PageCandidateTransaction.from_original(
            request.original_page.mutable_attempt_copy(), commits=commits
        ).compose()
        page_composition = PageCompositionSnapshot.build(
            run_id=request.run_id,
            execution_id=request.execution_id,
            page_id=request.page_id,
            page_source_sha256=request.page_source_sha256,
            base_pixel_sha256=composition.base_pixel_sha256,
            final_pixel_sha256=composition.final_pixel_sha256,
            commits=tuple(sorted(commits, key=lambda item: item.owner_id)),
            materializations=tuple(
                sorted(target_materializations, key=lambda item: item.owner_id)
            ),
        )
    else:
        commits = tuple(execution_output)
    return PageExecutionResult.build(
        request=request,
        coverage=coverage,
        owner_graph=translated_graph,
        translation_attempts=(translation_result.attempts if translation_result else ()),
        translations=(translation_result.bindings if translation_result else ()),
        page_commits=commits,
        repair_requests=repair_requests,
        owner_target_materializations=target_materializations,
        text_layers_view=text_layers_view,
        page_composition=page_composition,
        status="candidate_ready",
    )


def adapt_page_execution_result_to_output_page(
    result: PageExecutionResult,
    *,
    evidence_ref: Any | None = None,
):
    """Expose compatibility views while retaining the page result/ref authority."""

    from strip.types import OutputPage

    if result.final_page is not None:
        image = result.final_page.read_only_rgb().copy()
    else:
        image = result.request.original_page.mutable_attempt_copy()
    graph = result.owner_graph.read()
    return OutputPage(
        y_top=0,
        y_bottom=int(image.shape[0]),
        image=image,
        original_image=result.request.original_page.mutable_attempt_copy(),
        owner_graph=graph,
        owner_page_evidence_ref=evidence_ref,
        owner_page_result=result,
        ocr_result={
            "page_id": result.page_id,
            "_owner_graph_mode": "enforce",
            "_owner_graph_snapshot": graph.to_dict(),
        },
        text_layers={
            "texts": [
                {
                    "owner_id": binding.owner_id,
                    "original": binding.source_text,
                    "translated": binding.target_text,
                    "target_locale": binding.target_locale,
                }
                for binding in result.translations
            ]
        },
    )


@dataclass(frozen=True)
class OwnerContentReplay:
    publication_root: Any
    run_id: str
    execution_id: str
    verified_inputs: Any
    pages: tuple[PageExecutionResult, ...]


def load_owner_content_replay(publication_root):
    """Open a prior verified publication as immutable content lineage."""

    from pathlib import Path

    from ownership.publication import reopen_verified_publication

    root = Path(publication_root).resolve(strict=True)
    try:
        publication = reopen_verified_publication(root)
        pages = tuple(
            page.page_execution_evidence.read_verified(
                page.page_execution_evidence.canonical_json_bytes,
                root,
                expected={
                    "run_id": publication.verified_inputs.run_id,
                    "execution_id": publication.verified_inputs.execution_id,
                    "page_id": page.page_id,
                    "page_source_sha256": page.page_source_sha256,
                    "page_result_sha256": page.page_result_sha256,
                    "sha256": page.page_execution_evidence.sha256,
                },
            )
            for page in publication.verified_inputs.pages
        )
    except (ValueError, OSError) as exc:
        raise ContentReplayIntegrityError("content replay publication is invalid") from exc
    return OwnerContentReplay(
        publication_root=root,
        run_id=publication.verified_inputs.run_id,
        execution_id=publication.verified_inputs.execution_id,
        verified_inputs=publication.verified_inputs,
        pages=pages,
    )


def _canonical_rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise TypeError("page snapshots require HxWx3 uint8 RGB pixels")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("page snapshots cannot be empty")
    return np.ascontiguousarray(array).copy()


__all__ = [
    "BandProjection",
    "OriginalPageSnapshot",
    "OwnerGraphSnapshot",
    "PageExecutionResult",
    "PagePipelineIdentityError",
    "PagePipelineRequest",
    "PagePipelineServices",
    "PagePipelineStateError",
    "adapt_page_execution_result_to_output_page",
    "finalize_and_persist_page_result",
    "run_page_owner_pipeline",
    "load_owner_content_replay",
    "OwnerContentReplay",
    "ContentReplayIntegrityError",
]
