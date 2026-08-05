"""Canonical page-first coordinator for enforce-mode owner processing."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from io import BytesIO
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from ownership.coverage import PageCoverageResult
from ownership.hash_contract import (
    canonical_json_bytes,
    canonical_page_sha256,
    sha256_bytes,
)
from ownership.model import TRANSLATION_ROUTE_ACTIONS, OwnerGraph, OwnerProjection
from ownership.owner_builder import build_owner_page_graph_from_coverage
from ownership.translation import (
    OwnerPageTranslationResult,
    OwnerTranslationRequest,
    TranslationAttempt,
    TranslationBinding,
    apply_owner_translation_result,
    translate_owner_page,
)


class PagePipelineIdentityError(ValueError):
    """Raised when one page result crosses content or execution identity."""


class PagePipelineStateError(ValueError):
    """Raised when a page result claims an unsupported lifecycle state."""


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

    @property
    def page_id(self) -> str:
        return self.request.page_id

    @property
    def lifecycle(self):
        return self.coverage.ledger

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
        return {
            "schema_version": 1,
            "run_id": self.request.run_id,
            "execution_id": self.request.execution_id,
            "replay_of_execution_id": self.request.replay_of_execution_id,
            "replay_source_page_evidence_sha256": None,
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
            "repair_requests": [],
            "repair_history": [],
            "repair_budget_policy_sha256": None,
            "owner_target_materializations": [],
            "page_commits": [
                {
                    key: getattr(item, key, None)
                    for key in (
                        "commit_id", "run_id", "execution_id", "page_id",
                        "page_source_sha256", "owner_id", "before_sha256", "after_sha256",
                    )
                }
                for item in self.page_commits
            ],
            "page_geometry": None,
            "ocr_result": None,
            "text_layers": None,
            "page_composition": None,
            "project_asset_refs": [],
            "visual_stage_artifacts": None,
            "final_qa_ocr_requests": [],
            "final_qa_ocr_invocations": [],
            "language_residual_issues": [],
            "qa_probes": [],
            "final_replacement_verdicts": [],
            "replacement_verification_policy_sha256": None,
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
            PersistedRGBImageArtifactRef,
            TerminalPixelProof,
        )

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
        result = cls.build(
            request=request,
            coverage=coverage,
            owner_graph=graph,
            translation_attempts=attempts,
            translations=bindings,
            page_commits=tuple(SimpleNamespace(**item) for item in payload.get("page_commits") or ()),
            status=str(payload.get("status") or ""),
            final_page=final_page,
            terminal_proof=terminal_proof,
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
        status: str = "candidate_ready",
        final_page: Any | None = None,
        terminal_proof: Any | None = None,
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
        if status not in {"candidate_ready", "final_verified"}:
            raise PagePipelineStateError("page result state is not exportable")
        if status == "candidate_ready" and (final_page is not None or terminal_proof is not None):
            raise PagePipelineStateError("candidate result cannot carry final authority")
        if status == "final_verified" and (final_page is None or terminal_proof is None):
            raise PagePipelineStateError("final result requires both final page and terminal proof")
        if status == "final_verified":
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
                item.translation_binding_sha256 for item in bindings
            ):
                raise PagePipelineIdentityError("terminal proof translation binding chain mismatch")
            if tuple(getattr(terminal_proof, "source_payload_sha256s", ())) != tuple(
                item.source_payload_sha256 for item in bindings
            ) or tuple(getattr(terminal_proof, "target_payload_sha256s", ())) != tuple(
                item.target_payload_sha256 for item in bindings
            ):
                raise PagePipelineIdentityError("terminal proof payload chain mismatch")
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
            "page_id": request.page_id,
            "page_source_sha256": request.page_source_sha256,
            "coverage_sha256": coverage.sha256,
            "owner_graph_sha256": graph_snapshot.sha256,
            "translation_attempt_sha256s": [item.attempt_sha256 for item in attempts],
            "translation_binding_sha256s": [item.translation_binding_sha256 for item in bindings],
            "commit_ids": [str(getattr(item, "commit_id", "")) for item in commits],
            "status": status,
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
        )


@dataclass(frozen=True)
class PagePipelineServices:
    coverage_fn: Callable[[PagePipelineRequest], PageCoverageResult]
    translation_backends: tuple[Callable[..., Any], ...]
    graph_fn: Callable[[PageCoverageResult], OwnerGraph] = build_owner_page_graph_from_coverage
    translation_attempt_fn: Callable[..., Any] | None = None
    translation_attempt_controls: tuple[Any, ...] = ()
    execution_fn: Callable[[PagePipelineRequest, OwnerGraph, OwnerPageTranslationResult], Sequence[Any]] | None = None


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
        translation_result = translate_owner_page(
            requests,
            backends=services.translation_backends,
            attempt_fn=services.translation_attempt_fn,
            attempt_controls=services.translation_attempt_controls,
        )
        translated_graph = apply_owner_translation_result(graph, translation_result)
    else:
        translation_result = None
        translated_graph = graph
    commits = tuple(
        services.execution_fn(request, translated_graph, translation_result)
        if services.execution_fn is not None
        else ()
    )
    return PageExecutionResult.build(
        request=request,
        coverage=coverage,
        owner_graph=translated_graph,
        translation_attempts=(translation_result.attempts if translation_result else ()),
        translations=(translation_result.bindings if translation_result else ()),
        page_commits=commits,
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
    "run_page_owner_pipeline",
]
