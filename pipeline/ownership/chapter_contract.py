"""Immutable chapter-level source lineage contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
from typing import Any, Literal, Mapping, Sequence

from PIL import Image

from .hash_contract import (
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_page_sha256,
    canonical_source_tree_sha256,
    sha256_bytes,
    sha256_file,
)


CHAPTER_SOURCE_MANIFEST_SCHEMA_VERSION = 1
VERIFIED_PROJECT_INPUTS_SCHEMA_VERSION = 1
CHAPTER_ASSET_MANIFEST_SCHEMA_VERSION = 1
EXPORT_MANIFEST_SCHEMA_VERSION = 1
PUBLICATION_RECEIPT_SCHEMA_VERSION = 1
VERIFIED_CHAPTER_BUNDLE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ChapterCardinalityError(ValueError):
    """Raised when source pages and verified page results are not an ordered bijection."""


def _require_sha256(value: str, field: str) -> str:
    normalized = str(value or "")
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return normalized


def _portable_relative_path(value: str) -> str:
    raw = str(value or "")
    if not raw or "\\" in raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"relative source path is not portable: {raw!r}")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"relative source path escapes its root: {raw!r}")
    normalized = path.as_posix()
    if normalized != raw:
        raise ValueError(f"relative source path is not canonical: {raw!r}")
    return normalized


@dataclass(frozen=True)
class SourcePageManifestEntry:
    page_id: str
    relative_source_path: str
    source_file_sha256: str
    page_source_sha256: str
    width: int
    height: int
    mode: Literal["RGB"] = "RGB"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"page_[0-9]{3,}", self.page_id):
            raise ValueError(f"invalid page_id: {self.page_id!r}")
        object.__setattr__(self, "relative_source_path", _portable_relative_path(self.relative_source_path))
        _require_sha256(self.source_file_sha256, "source_file_sha256")
        _require_sha256(self.page_source_sha256, "page_source_sha256")
        if self.width <= 0 or self.height <= 0 or self.mode != "RGB":
            raise ValueError("source page metadata must describe a non-empty RGB image")

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "relative_source_path": self.relative_source_path,
            "source_file_sha256": self.source_file_sha256,
            "page_source_sha256": self.page_source_sha256,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourcePageManifestEntry":
        expected = {
            "page_id", "relative_source_path", "source_file_sha256", "page_source_sha256",
            "width", "height", "mode",
        }
        if set(value) != expected:
            raise ValueError("source page entry contains missing or unknown fields")
        return cls(
            page_id=str(value["page_id"]),
            relative_source_path=str(value["relative_source_path"]),
            source_file_sha256=str(value["source_file_sha256"]),
            page_source_sha256=str(value["page_source_sha256"]),
            width=int(value["width"]),
            height=int(value["height"]),
            mode=str(value["mode"]),
        )


@dataclass(frozen=True)
class ChapterSourceManifest:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    pages: tuple[SourcePageManifestEntry, ...]
    source_page_count: int
    source_tree_sha256: str
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def _build(
        cls,
        *,
        run_id: str,
        execution_id: str,
        replay_of_execution_id: str | None,
        pages: Sequence[SourcePageManifestEntry],
        source_tree_sha256: str,
    ) -> "ChapterSourceManifest":
        page_tuple = tuple(pages)
        if not run_id or not execution_id or not page_tuple:
            raise ValueError("chapter source manifest requires identity and at least one page")
        expected_ids = tuple(f"page_{index:03d}" for index in range(1, len(page_tuple) + 1))
        if tuple(page.page_id for page in page_tuple) != expected_ids:
            raise ValueError("source page order and page_id sequence disagree")
        paths = tuple(page.relative_source_path for page in page_tuple)
        folded = tuple(unicodedata.normalize("NFC", path).casefold() for path in paths)
        if len(set(paths)) != len(paths) or len(set(folded)) != len(folded):
            raise ValueError("source page relative path collision")
        expected_tree = canonical_json_sha256(
            {
                "entries": [
                    {
                        "height": page.height,
                        "mode": page.mode,
                        "page_source_sha256": page.page_source_sha256,
                        "relative_path": page.relative_source_path,
                        "source_file_sha256": page.source_file_sha256,
                        "width": page.width,
                    }
                    for page in page_tuple
                ],
                "schema_version": 1,
            }
        )
        _require_sha256(source_tree_sha256, "source_tree_sha256")
        if expected_tree != source_tree_sha256:
            raise ValueError("source tree hash does not match page order or metadata")
        payload = {
            "schema_version": CHAPTER_SOURCE_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": execution_id,
            "replay_of_execution_id": replay_of_execution_id,
            "pages": [page.to_dict() for page in page_tuple],
            "source_page_count": len(page_tuple),
            "source_tree_sha256": source_tree_sha256,
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            schema_version=CHAPTER_SOURCE_MANIFEST_SCHEMA_VERSION,
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
            pages=page_tuple,
            source_page_count=len(page_tuple),
            source_tree_sha256=source_tree_sha256,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    @classmethod
    def from_extracted_pages(
        cls,
        image_files: Sequence[str | Path],
        extraction_root: str | Path,
        *,
        run_id: str,
        execution_id: str,
        replay_of_execution_id: str | None = None,
    ) -> "ChapterSourceManifest":
        root = Path(extraction_root).resolve(strict=True)
        paths = tuple(Path(path).resolve(strict=True) for path in image_files)
        pages: list[SourcePageManifestEntry] = []
        for index, path in enumerate(paths, 1):
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"source page is outside extraction root: {path}") from exc
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
                width, height = rgb.size
                pixel_sha256 = canonical_page_sha256(rgb)
            pages.append(
                SourcePageManifestEntry(
                    page_id=f"page_{index:03d}",
                    relative_source_path=relative,
                    source_file_sha256=sha256_file(path),
                    page_source_sha256=pixel_sha256,
                    width=width,
                    height=height,
                )
            )
        return cls._build(
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
            pages=pages,
            source_tree_sha256=canonical_source_tree_sha256(paths, root),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json_bytes.decode("utf-8"))

    @classmethod
    def from_canonical_json_bytes(cls, value: bytes) -> "ChapterSourceManifest":
        try:
            payload = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid chapter source manifest JSON") from exc
        expected = {
            "schema_version", "run_id", "execution_id", "replay_of_execution_id",
            "pages", "source_page_count", "source_tree_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("chapter source manifest contains missing or unknown fields")
        if payload["schema_version"] != CHAPTER_SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unknown chapter source manifest schema version")
        pages = tuple(SourcePageManifestEntry.from_dict(item) for item in payload["pages"])
        if int(payload["source_page_count"]) != len(pages):
            raise ValueError("source page count does not match pages")
        rebuilt = cls._build(
            run_id=str(payload["run_id"]),
            execution_id=str(payload["execution_id"]),
            replay_of_execution_id=(
                str(payload["replay_of_execution_id"])
                if payload["replay_of_execution_id"] is not None else None
            ),
            pages=pages,
            source_tree_sha256=str(payload["source_tree_sha256"]),
        )
        if rebuilt.canonical_json_bytes != value:
            raise ValueError("chapter source manifest JSON is not canonical")
        return rebuilt


@dataclass(frozen=True)
class VerifiedPageProjectInput:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_content_semantic_sha256: str
    artifact_store_id: str
    generation_id: str
    page_generation_id: str
    page_execution_evidence_relative_path: str
    page_execution_evidence: Any
    original_artifact: Any
    final_artifact: Any
    owner_graph: Any
    terminal_proof_sha256: str

    @classmethod
    def from_verified_result(cls, result: Any, evidence_ref: Any) -> "VerifiedPageProjectInput":
        if result.status != "final_verified" or result.final_page is None or result.terminal_proof is None:
            raise ValueError("verified project page requires terminal page authority")
        if (
            evidence_ref.run_id,
            evidence_ref.execution_id,
            evidence_ref.page_id,
            evidence_ref.page_source_sha256,
        ) != (
            result.request.run_id,
            result.request.execution_id,
            result.page_id,
            result.request.page_source_sha256,
        ):
            raise ValueError("page evidence ref differs from reopened result")
        from .execution import PageExecutionEvidenceSnapshot

        snapshot = PageExecutionEvidenceSnapshot.build(result)
        if snapshot.sha256 != evidence_ref.page_execution_evidence_sha256:
            raise ValueError("page evidence snapshot hash differs from pointer")
        return cls(
            run_id=result.request.run_id,
            execution_id=result.request.execution_id,
            replay_of_execution_id=result.request.replay_of_execution_id,
            page_id=result.page_id,
            page_source_sha256=result.request.page_source_sha256,
            page_result_sha256=result.result_sha256,
            page_content_semantic_sha256=result.page_content_semantic_sha256,
            artifact_store_id=evidence_ref.artifact_store_id,
            generation_id=evidence_ref.generation_id,
            page_generation_id=evidence_ref.page_generation_id,
            page_execution_evidence_relative_path=evidence_ref.page_execution_evidence_relative_path,
            page_execution_evidence=snapshot,
            original_artifact=result.request.original_page.artifact_ref,
            final_artifact=result.final_page.artifact_ref,
            owner_graph=result.owner_graph,
            terminal_proof_sha256=result.terminal_proof.proof_sha256,
        )

    def canonical_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "replay_of_execution_id": self.replay_of_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "page_result_sha256": self.page_result_sha256,
            "page_content_semantic_sha256": self.page_content_semantic_sha256,
            "artifact_store_id": self.artifact_store_id,
            "generation_id": self.generation_id,
            "page_generation_id": self.page_generation_id,
            "page_execution_evidence_relative_path": self.page_execution_evidence_relative_path,
            "page_execution_evidence_sha256": self.page_execution_evidence.sha256,
            "original_artifact_ref_sha256": self.original_artifact.artifact_ref_sha256,
            "final_artifact_ref_sha256": self.final_artifact.artifact_ref_sha256,
            "owner_graph_sha256": self.owner_graph.sha256,
            "terminal_proof_sha256": self.terminal_proof_sha256,
        }


@dataclass(frozen=True)
class VerifiedProjectInputs:
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    source_manifest: ChapterSourceManifest
    source_manifest_sha256: str
    artifact_store_id: str
    generation_id: str
    pages: tuple[VerifiedPageProjectInput, ...]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(
        cls,
        source_manifest: ChapterSourceManifest,
        pages: Sequence[VerifiedPageProjectInput],
    ) -> "VerifiedProjectInputs":
        page_tuple = tuple(pages)
        expected = tuple((page.page_id, page.page_source_sha256) for page in source_manifest.pages)
        actual = tuple((page.page_id, page.page_source_sha256) for page in page_tuple)
        if actual != expected:
            raise ChapterCardinalityError("verified pages are not an ordered bijection with source manifest")
        identities = {
            (
                page.run_id, page.execution_id, page.replay_of_execution_id,
                page.artifact_store_id, page.generation_id,
            )
            for page in page_tuple
        }
        expected_identity = (
            source_manifest.run_id,
            source_manifest.execution_id,
            source_manifest.replay_of_execution_id,
        )
        if len(identities) != 1:
            raise ChapterCardinalityError("verified pages mix execution generations")
        run_id, execution_id, replay_id, store_id, generation_id = next(iter(identities))
        if (run_id, execution_id, replay_id) != expected_identity:
            raise ChapterCardinalityError("verified page identity differs from source manifest")
        generation_ids = [page.page_generation_id for page in page_tuple]
        evidence_paths = [page.page_execution_evidence_relative_path for page in page_tuple]
        if len(generation_ids) != len(set(generation_ids)) or len(evidence_paths) != len(set(evidence_paths)):
            raise ChapterCardinalityError("verified pages reuse page generation or evidence path")
        payload = {
            "schema_version": VERIFIED_PROJECT_INPUTS_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": execution_id,
            "replay_of_execution_id": replay_id,
            "source_manifest_sha256": source_manifest.sha256,
            "artifact_store_id": store_id,
            "generation_id": generation_id,
            "pages": [page.canonical_summary() for page in page_tuple],
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_id,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest.sha256,
            artifact_store_id=store_id,
            generation_id=generation_id,
            pages=page_tuple,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    @classmethod
    def read_verified(cls, generation_root: str | Path) -> "VerifiedProjectInputs":
        """Reopen the frozen project boundary without consulting live OutputPage views."""

        from .execution import (
            ArtifactGenerationMarker,
            PageArtifactIntegrityError,
            PageExecutionEvidenceSnapshot,
        )

        root = Path(generation_root).resolve(strict=True)
        marker = ArtifactGenerationMarker.read_verified(root)
        source_path = root / "chapter_source_manifest.json"
        inputs_path = root / "evidence" / "verified_project_inputs.json"
        try:
            source_bytes = source_path.read_bytes()
            encoded = inputs_path.read_bytes()
        except OSError as exc:
            raise PageArtifactIntegrityError("verified project controls are missing") from exc
        source_manifest = ChapterSourceManifest.from_canonical_json_bytes(source_bytes)
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PageArtifactIntegrityError("verified project inputs JSON is invalid") from exc
        expected_top = {
            "schema_version", "run_id", "execution_id", "replay_of_execution_id",
            "source_manifest_sha256", "artifact_store_id", "generation_id", "pages",
        }
        if not isinstance(payload, dict) or set(payload) != expected_top:
            raise PageArtifactIntegrityError(
                "verified project inputs contain missing or unknown fields"
            )
        if payload["schema_version"] != VERIFIED_PROJECT_INPUTS_SCHEMA_VERSION:
            raise PageArtifactIntegrityError("verified project inputs schema is unsupported")
        identity = (
            str(payload["run_id"]), str(payload["execution_id"]),
            payload["replay_of_execution_id"], str(payload["artifact_store_id"]),
            str(payload["generation_id"]),
        )
        if identity != (
            marker.run_id, marker.execution_id, source_manifest.replay_of_execution_id,
            marker.artifact_store_id, marker.generation_id,
        ):
            raise PageArtifactIntegrityError("verified project generation identity mismatch")
        if str(payload["source_manifest_sha256"]) != source_manifest.sha256:
            raise PageArtifactIntegrityError("verified project source manifest hash mismatch")

        expected_page_fields = {
            "run_id", "execution_id", "replay_of_execution_id", "page_id",
            "page_source_sha256", "page_result_sha256",
            "page_content_semantic_sha256", "artifact_store_id", "generation_id",
            "page_generation_id", "page_execution_evidence_relative_path",
            "page_execution_evidence_sha256", "original_artifact_ref_sha256",
            "final_artifact_ref_sha256", "owner_graph_sha256", "terminal_proof_sha256",
        }
        pages: list[VerifiedPageProjectInput] = []
        for summary in payload["pages"]:
            if not isinstance(summary, dict) or set(summary) != expected_page_fields:
                raise PageArtifactIntegrityError(
                    "verified project page contains missing or unknown fields"
                )
            relative = _portable_relative_path(
                str(summary["page_execution_evidence_relative_path"])
            )
            evidence_path = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
            try:
                evidence_path.relative_to(root)
            except ValueError as exc:
                raise PageArtifactIntegrityError("page evidence escapes generation root") from exc
            evidence_bytes = evidence_path.read_bytes()
            if sha256_bytes(evidence_bytes) != str(summary["page_execution_evidence_sha256"]):
                raise PageArtifactIntegrityError("page evidence file hash mismatch")
            result = PageExecutionEvidenceSnapshot.read_verified(
                evidence_bytes,
                root,
                expected={
                    "run_id": marker.run_id,
                    "execution_id": marker.execution_id,
                    "page_id": str(summary["page_id"]),
                    "page_source_sha256": str(summary["page_source_sha256"]),
                    "artifact_store_id": marker.artifact_store_id,
                    "generation_id": marker.generation_id,
                    "sha256": str(summary["page_execution_evidence_sha256"]),
                    "page_result_sha256": str(summary["page_result_sha256"]),
                },
            )
            snapshot = PageExecutionEvidenceSnapshot.build(result)
            page = VerifiedPageProjectInput(
                run_id=result.request.run_id,
                execution_id=result.request.execution_id,
                replay_of_execution_id=result.request.replay_of_execution_id,
                page_id=result.page_id,
                page_source_sha256=result.request.page_source_sha256,
                page_result_sha256=result.result_sha256,
                page_content_semantic_sha256=result.page_content_semantic_sha256,
                artifact_store_id=marker.artifact_store_id,
                generation_id=marker.generation_id,
                page_generation_id=str(summary["page_generation_id"]),
                page_execution_evidence_relative_path=relative,
                page_execution_evidence=snapshot,
                original_artifact=result.request.original_page.artifact_ref,
                final_artifact=result.final_page.artifact_ref,
                owner_graph=result.owner_graph,
                terminal_proof_sha256=result.terminal_proof.proof_sha256,
            )
            if page.canonical_summary() != summary:
                raise PageArtifactIntegrityError("verified project page summary mismatch")
            pages.append(page)
        rebuilt = cls.build(source_manifest, pages)
        if rebuilt.canonical_json_bytes != encoded:
            raise PageArtifactIntegrityError("verified project inputs are not canonical")
        return rebuilt


@dataclass(frozen=True)
class ChapterAssetEntry:
    relative_path: str
    kind: str
    project_reference: str | None
    evidence_reference: str | None
    file_sha256: str
    media_type: str
    decoded_mode: str | None
    decoded_pixel_sha256: str | None
    width: int | None
    height: int | None
    size_bytes: int
    page_id: str | None
    owner_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "project_reference": self.project_reference,
            "evidence_reference": self.evidence_reference,
            "file_sha256": self.file_sha256,
            "media_type": self.media_type,
            "decoded_mode": self.decoded_mode,
            "decoded_pixel_sha256": self.decoded_pixel_sha256,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "page_id": self.page_id,
            "owner_ids": list(self.owner_ids),
        }


@dataclass(frozen=True)
class ChapterAssetManifest:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    entries: tuple[ChapterAssetEntry, ...]
    asset_tree_sha256: str
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(
        cls,
        generation_root: str | Path,
        *,
        run_id: str,
        execution_id: str,
        replay_of_execution_id: str | None,
        artifact_store_id: str,
        generation_id: str,
    ) -> "ChapterAssetManifest":
        root = Path(generation_root).resolve(strict=True)
        excluded = {
            "chapter_source_manifest.json", "chapter_asset_manifest.json",
            "export_manifest.json", "publication_receipt.json",
        }
        entries: list[ChapterAssetEntry] = []
        folded: set[str] = set()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.is_symlink():
                raise ValueError("asset tree cannot contain symlinks")
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            portable = _portable_relative_path(relative)
            key = unicodedata.normalize("NFC", portable).casefold()
            if key in folded:
                raise ValueError("asset tree contains a portable path collision")
            folded.add(key)
            media_type = "application/json" if path.suffix.casefold() == ".json" else "application/octet-stream"
            decoded_mode = None
            decoded_pixel_sha256 = None
            width = None
            height = None
            if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
                with Image.open(path) as opened:
                    decoded = opened.convert("RGB")
                    width, height = decoded.size
                    decoded_mode = "RGB"
                    decoded_pixel_sha256 = canonical_page_sha256(decoded)
                media_type = "image/png" if path.suffix.casefold() == ".png" else f"image/{path.suffix.casefold().lstrip('.')}"
            page_match = re.search(r"(page_[0-9]{3,})", portable)
            kind = (
                "project" if portable == "project.json" else
                "verified_inputs" if portable == "evidence/verified_project_inputs.json" else
                "page_evidence" if portable.endswith("page_execution_evidence.json") else
                "final_page" if portable.startswith("translated/") else
                "source_page" if portable.startswith("originals/") else
                "generation_control" if portable == "artifact_generation.json" else
                "schema_asset"
            )
            entries.append(ChapterAssetEntry(
                relative_path=portable,
                kind=kind,
                project_reference=portable if portable == "project.json" or portable.startswith(("originals/", "translated/", "layers/", "images/")) else None,
                evidence_reference=portable if portable.startswith(("evidence/", ".page-generations/", ".page-current/")) else None,
                file_sha256=sha256_file(path),
                media_type=media_type,
                decoded_mode=decoded_mode,
                decoded_pixel_sha256=decoded_pixel_sha256,
                width=width,
                height=height,
                size_bytes=path.stat().st_size,
                page_id=page_match.group(1) if page_match else None,
            ))
        entry_payloads = [entry.to_dict() for entry in entries]
        tree_sha = canonical_json_sha256({"schema_version": 1, "entries": entry_payloads})
        payload = {
            "schema_version": CHAPTER_ASSET_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": execution_id,
            "replay_of_execution_id": replay_of_execution_id,
            "artifact_store_id": artifact_store_id,
            "generation_id": generation_id,
            "entries": entry_payloads,
            "asset_tree_sha256": tree_sha,
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            schema_version=CHAPTER_ASSET_MANIFEST_SCHEMA_VERSION,
            run_id=run_id,
            execution_id=execution_id,
            replay_of_execution_id=replay_of_execution_id,
            artifact_store_id=artifact_store_id,
            generation_id=generation_id,
            entries=tuple(entries),
            asset_tree_sha256=tree_sha,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )


@dataclass(frozen=True)
class ExportPageEntry:
    page_id: str
    page_source_sha256: str
    page_result_sha256: str
    page_execution_evidence_sha256: str
    page_content_semantic_sha256: str
    final_pixel_sha256: str
    translated_path: str
    entry_sha256: str

    @classmethod
    def build(cls, page: VerifiedPageProjectInput) -> "ExportPageEntry":
        payload = {
            "page_id": page.page_id,
            "page_source_sha256": page.page_source_sha256,
            "page_result_sha256": page.page_result_sha256,
            "page_execution_evidence_sha256": page.page_execution_evidence.sha256,
            "page_content_semantic_sha256": page.page_content_semantic_sha256,
            "final_pixel_sha256": page.final_artifact.pixel_sha256,
            "translated_path": page.final_artifact.relative_path,
        }
        return cls(**payload, entry_sha256=canonical_json_sha256(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "page_result_sha256": self.page_result_sha256,
            "page_execution_evidence_sha256": self.page_execution_evidence_sha256,
            "page_content_semantic_sha256": self.page_content_semantic_sha256,
            "final_pixel_sha256": self.final_pixel_sha256,
            "translated_path": self.translated_path,
            "entry_sha256": self.entry_sha256,
        }


@dataclass(frozen=True)
class ExportManifest:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    source_manifest_sha256: str
    verified_inputs_sha256: str
    asset_manifest_sha256: str
    replay_source_page_evidence_sha256s: tuple[str, ...]
    pages: tuple[ExportPageEntry, ...]
    export_gate_status: Literal["PASS"]
    canonical_json_bytes: bytes
    sha256: str

    @classmethod
    def build(cls, inputs: VerifiedProjectInputs, assets: ChapterAssetManifest) -> "ExportManifest":
        pages = tuple(ExportPageEntry.build(page) for page in inputs.pages)
        payload = {
            "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            "run_id": inputs.run_id,
            "execution_id": inputs.execution_id,
            "replay_of_execution_id": inputs.replay_of_execution_id,
            "artifact_store_id": inputs.artifact_store_id,
            "generation_id": inputs.generation_id,
            "source_manifest_sha256": inputs.source_manifest_sha256,
            "verified_inputs_sha256": inputs.sha256,
            "asset_manifest_sha256": assets.sha256,
            "replay_source_page_evidence_sha256s": [],
            "pages": [page.to_dict() for page in pages],
            "export_gate_status": "PASS",
        }
        encoded = canonical_json_bytes(payload)
        return cls(
            schema_version=EXPORT_MANIFEST_SCHEMA_VERSION,
            run_id=inputs.run_id,
            execution_id=inputs.execution_id,
            replay_of_execution_id=inputs.replay_of_execution_id,
            artifact_store_id=inputs.artifact_store_id,
            generation_id=inputs.generation_id,
            source_manifest_sha256=inputs.source_manifest_sha256,
            verified_inputs_sha256=inputs.sha256,
            asset_manifest_sha256=assets.sha256,
            replay_source_page_evidence_sha256s=(),
            pages=pages,
            export_gate_status="PASS",
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )


@dataclass(frozen=True)
class PublicationReceipt:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    source_manifest_sha256: str
    asset_manifest_sha256: str
    verified_inputs_sha256: str
    export_manifest_sha256: str
    payload_tree_sha256: str
    receipt_sha256: str
    canonical_json_bytes: bytes

    @classmethod
    def build(
        cls, inputs: VerifiedProjectInputs, assets: ChapterAssetManifest,
        export: ExportManifest,
    ) -> "PublicationReceipt":
        payload = {
            "schema_version": PUBLICATION_RECEIPT_SCHEMA_VERSION,
            "run_id": inputs.run_id,
            "execution_id": inputs.execution_id,
            "replay_of_execution_id": inputs.replay_of_execution_id,
            "artifact_store_id": inputs.artifact_store_id,
            "generation_id": inputs.generation_id,
            "source_manifest_sha256": inputs.source_manifest_sha256,
            "asset_manifest_sha256": assets.sha256,
            "verified_inputs_sha256": inputs.sha256,
            "export_manifest_sha256": export.sha256,
            "payload_tree_sha256": assets.asset_tree_sha256,
        }
        receipt_sha = canonical_json_sha256(payload)
        encoded = canonical_json_bytes(payload | {"receipt_sha256": receipt_sha})
        return cls(**payload, receipt_sha256=receipt_sha, canonical_json_bytes=encoded)


@dataclass(frozen=True)
class VerifiedChapterBundle:
    schema_version: int
    run_id: str
    execution_id: str
    replay_of_execution_id: str | None
    artifact_store_id: str
    generation_id: str
    source_manifest_sha256: str
    verified_inputs_sha256: str
    asset_manifest: ChapterAssetManifest
    export_manifest: ExportManifest
    publication_receipt: PublicationReceipt
    generation_tree_sha256: str
    bundle_sha256: str
    runtime_staging_root: Path

    @classmethod
    def build(
        cls,
        *,
        runtime_staging_root: str | Path,
        inputs: VerifiedProjectInputs,
        asset_manifest: ChapterAssetManifest,
        export_manifest: ExportManifest,
        publication_receipt: PublicationReceipt,
    ) -> "VerifiedChapterBundle":
        payload = {
            "schema_version": VERIFIED_CHAPTER_BUNDLE_SCHEMA_VERSION,
            "run_id": inputs.run_id,
            "execution_id": inputs.execution_id,
            "replay_of_execution_id": inputs.replay_of_execution_id,
            "artifact_store_id": inputs.artifact_store_id,
            "generation_id": inputs.generation_id,
            "source_manifest_sha256": inputs.source_manifest_sha256,
            "verified_inputs_sha256": inputs.sha256,
            "asset_manifest_sha256": asset_manifest.sha256,
            "export_manifest_sha256": export_manifest.sha256,
            "publication_receipt_sha256": publication_receipt.receipt_sha256,
            "generation_tree_sha256": asset_manifest.asset_tree_sha256,
        }
        return cls(
            schema_version=VERIFIED_CHAPTER_BUNDLE_SCHEMA_VERSION,
            run_id=inputs.run_id,
            execution_id=inputs.execution_id,
            replay_of_execution_id=inputs.replay_of_execution_id,
            artifact_store_id=inputs.artifact_store_id,
            generation_id=inputs.generation_id,
            source_manifest_sha256=inputs.source_manifest_sha256,
            verified_inputs_sha256=inputs.sha256,
            asset_manifest=asset_manifest,
            export_manifest=export_manifest,
            publication_receipt=publication_receipt,
            generation_tree_sha256=asset_manifest.asset_tree_sha256,
            bundle_sha256=canonical_json_sha256(payload),
            runtime_staging_root=Path(runtime_staging_root).resolve(strict=True),
        )
