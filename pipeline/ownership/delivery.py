"""Hash-bound proof that one complete translated owner body reached glyph pixels."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


BBox = tuple[int, int, int, int]


def normalize_delivery_payload(value: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(normalized.replace("\r", "\n").split())


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"traduzai.ndarray.v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_mask(value: Any, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.uint8 or raw.ndim != 2:
        raise ValueError("glyph core mask must be a uint8 2D array")
    if shape is not None and raw.shape != shape:
        raise ValueError("glyph span core mask shape mismatch")
    if not np.all((raw == 0) | (raw == 255)):
        raise ValueError("glyph core mask must be binary")
    return np.ascontiguousarray(raw)


def _mask_bbox(mask: np.ndarray) -> BBox | None:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class OwnerTextExecutionAuthority:
    schema_version: int
    owner_id: str
    page_id: str
    source_payload: str
    translated_payload: str
    normalized_chunks: tuple[str, ...]
    source_payload_sha256: str
    translated_payload_sha256: str
    normalized_chunks_sha256: str
    authority_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "source_payload": self.source_payload,
            "translated_payload": self.translated_payload,
            "normalized_chunks": list(self.normalized_chunks),
            "source_payload_sha256": self.source_payload_sha256,
            "translated_payload_sha256": self.translated_payload_sha256,
            "normalized_chunks_sha256": self.normalized_chunks_sha256,
            "authority_sha256": self.authority_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OwnerTextExecutionAuthority":
        if not isinstance(value, Mapping):
            raise TypeError("text execution authority must be a mapping")
        return seal_owner_text_execution_authority(
            owner_id=str(value.get("owner_id") or ""),
            page_id=str(value.get("page_id") or ""),
            source_payload=str(value.get("source_payload") or ""),
            translated_payload=str(value.get("translated_payload") or ""),
            normalized_chunks=tuple(value.get("normalized_chunks") or ()),
            expected_authority_sha256=str(value.get("authority_sha256") or ""),
        )


def seal_owner_text_execution_authority(
    *,
    owner_id: str,
    page_id: str,
    source_payload: str,
    translated_payload: str,
    normalized_chunks: Sequence[str],
    expected_authority_sha256: str | None = None,
) -> OwnerTextExecutionAuthority:
    owner_id = str(owner_id).strip()
    page_id = str(page_id).strip()
    if not owner_id or not page_id:
        raise ValueError("text execution authority requires owner and page identity")
    source = normalize_delivery_payload(source_payload)
    translated = normalize_delivery_payload(translated_payload)
    chunks = tuple(
        chunk for raw in normalized_chunks
        if (chunk := normalize_delivery_payload(raw))
    )
    if not source or not translated or not chunks:
        raise ValueError("text execution authority requires complete payloads")
    if normalize_delivery_payload(" ".join(chunks)) != translated:
        raise ValueError("normalized chunks do not reconstruct translated payload")
    chunks_hash = _canonical_hash({"chunks": list(chunks)})
    core = {
        "schema_version": 1,
        "owner_id": owner_id,
        "page_id": page_id,
        "source_payload_sha256": _sha_text(source),
        "translated_payload_sha256": _sha_text(translated),
        "normalized_chunks_sha256": chunks_hash,
    }
    authority_hash = _canonical_hash(core)
    if expected_authority_sha256 and expected_authority_sha256 != authority_hash:
        raise ValueError("text execution authority hash mismatch")
    return OwnerTextExecutionAuthority(
        **core,
        source_payload=source,
        translated_payload=translated,
        normalized_chunks=chunks,
        authority_sha256=authority_hash,
    )


@dataclass(frozen=True)
class GlyphRunObservation:
    span_index: int
    observed_text: str
    codepoints: tuple[int, ...]
    clusters: tuple[str, ...]
    font_identity: str
    font_run_sha256: str
    glyph_run_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = vars(self).copy()
        payload["codepoints"] = list(self.codepoints)
        payload["clusters"] = list(self.clusters)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlyphRunObservation":
        return cls(
            span_index=int(value["span_index"]),
            observed_text=str(value["observed_text"]),
            codepoints=tuple(int(item) for item in value.get("codepoints") or ()),
            clusters=tuple(str(item) for item in value.get("clusters") or ()),
            font_identity=str(value["font_identity"]),
            font_run_sha256=str(value["font_run_sha256"]),
            glyph_run_sha256=str(value["glyph_run_sha256"]),
        )

    @classmethod
    def build(
        cls, *, text: str, font_identity: str, span_index: int
    ) -> "GlyphRunObservation":
        observed = normalize_delivery_payload(text)
        clusters = tuple(character for character in observed if not character.isspace())
        codepoints = tuple(ord(character) for character in clusters)
        font_hash = _canonical_hash({"font_identity": str(font_identity)})
        payload = {
            "span_index": int(span_index),
            "observed_text": observed,
            "codepoints": list(codepoints),
            "clusters": list(clusters),
            "font_run_sha256": font_hash,
        }
        return cls(
            span_index=int(span_index),
            observed_text=observed,
            codepoints=codepoints,
            clusters=clusters,
            font_identity=str(font_identity),
            font_run_sha256=font_hash,
            glyph_run_sha256=_canonical_hash(payload),
        )


@dataclass(frozen=True)
class RenderedGlyphSpanEvidence:
    span_index: int
    text_start: int
    text_end: int
    normalized_text_sha256: str
    non_whitespace_glyph_count: int
    glyph_coverage_ratio: float
    font_run_sha256: str
    glyph_run_sha256: str
    core_mask_sha256: str
    core_mask_bbox_logical: BBox

    def to_dict(self) -> dict[str, Any]:
        payload = vars(self).copy()
        payload["core_mask_bbox_logical"] = list(self.core_mask_bbox_logical)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RenderedGlyphSpanEvidence":
        return cls(
            span_index=int(value["span_index"]),
            text_start=int(value["text_start"]),
            text_end=int(value["text_end"]),
            normalized_text_sha256=str(value["normalized_text_sha256"]),
            non_whitespace_glyph_count=int(value["non_whitespace_glyph_count"]),
            glyph_coverage_ratio=float(value["glyph_coverage_ratio"]),
            font_run_sha256=str(value["font_run_sha256"]),
            glyph_run_sha256=str(value["glyph_run_sha256"]),
            core_mask_sha256=str(value["core_mask_sha256"]),
            core_mask_bbox_logical=tuple(int(item) for item in value["core_mask_bbox_logical"]),
        )


@dataclass(frozen=True)
class OwnerTextDeliveryContract:
    schema_version: int
    owner_id: str
    page_id: str
    execution_authority_sha256: str
    source_payload_sha256: str
    translated_payload_sha256: str
    layout_payload_sha256: str
    rendered_payload_sha256: str
    rendered_patch_sha256: str
    rendered_line_count: int
    glyph_spans: tuple[RenderedGlyphSpanEvidence, ...]
    glyph_spans_sha256: str
    glyph_core_mask_sha256: str
    status: str
    reason: str
    contract_sha256: str

    def _core_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_id": self.owner_id,
            "page_id": self.page_id,
            "execution_authority_sha256": self.execution_authority_sha256,
            "source_payload_sha256": self.source_payload_sha256,
            "translated_payload_sha256": self.translated_payload_sha256,
            "layout_payload_sha256": self.layout_payload_sha256,
            "rendered_payload_sha256": self.rendered_payload_sha256,
            "rendered_patch_sha256": self.rendered_patch_sha256,
            "rendered_line_count": self.rendered_line_count,
            "glyph_spans": [span.to_dict() for span in self.glyph_spans],
            "glyph_spans_sha256": self.glyph_spans_sha256,
            "glyph_core_mask_sha256": self.glyph_core_mask_sha256,
            "status": self.status,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._core_dict(), "contract_sha256": self.contract_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OwnerTextDeliveryContract":
        return cls(
            schema_version=int(value["schema_version"]),
            owner_id=str(value["owner_id"]),
            page_id=str(value["page_id"]),
            execution_authority_sha256=str(value["execution_authority_sha256"]),
            source_payload_sha256=str(value["source_payload_sha256"]),
            translated_payload_sha256=str(value["translated_payload_sha256"]),
            layout_payload_sha256=str(value["layout_payload_sha256"]),
            rendered_payload_sha256=str(value["rendered_payload_sha256"]),
            rendered_patch_sha256=str(value["rendered_patch_sha256"]),
            rendered_line_count=int(value["rendered_line_count"]),
            glyph_spans=tuple(
                RenderedGlyphSpanEvidence.from_dict(item)
                for item in value.get("glyph_spans") or ()
            ),
            glyph_spans_sha256=str(value["glyph_spans_sha256"]),
            glyph_core_mask_sha256=str(value["glyph_core_mask_sha256"]),
            status=str(value["status"]),
            reason=str(value["reason"]),
            contract_sha256=str(value["contract_sha256"]),
        )


def _review_reason(authority_text: str, rendered_text: str) -> str:
    if rendered_text == authority_text:
        return "delivered"
    authorized_tokens = authority_text.split()
    rendered_tokens = rendered_text.split()
    if len(rendered_tokens) > len(authorized_tokens) or any(
        rendered_tokens.count(token) > authorized_tokens.count(token)
        for token in set(rendered_tokens)
    ):
        return "render_payload_duplicated"
    return "render_payload_incomplete"


def build_owner_text_delivery_contract(
    *,
    execution_authority: OwnerTextExecutionAuthority,
    layout_payload: str,
    rendered_lines: Sequence[str],
    rendered_glyph_runs: Sequence[GlyphRunObservation],
    glyph_core_mask: np.ndarray,
    glyph_span_core_masks: Sequence[np.ndarray] | None = None,
    rendered_patch_sha256: str,
) -> OwnerTextDeliveryContract:
    if not isinstance(execution_authority, OwnerTextExecutionAuthority):
        raise TypeError("delivery requires sealed text execution authority")
    authority_text = execution_authority.translated_payload
    layout_text = normalize_delivery_payload(layout_payload)
    lines = tuple(normalize_delivery_payload(line) for line in rendered_lines if normalize_delivery_payload(line))
    rendered_text = normalize_delivery_payload(" ".join(lines))
    core = _canonical_mask(glyph_core_mask)
    span_masks = tuple(
        _canonical_mask(mask, shape=core.shape)
        for mask in (glyph_span_core_masks or ((core,) if len(lines) == 1 else ()))
    )
    runs = tuple(rendered_glyph_runs)
    reason = _review_reason(authority_text, rendered_text)
    def evidence_failure(value: str) -> None:
        nonlocal reason
        if reason == "delivered":
            reason = value

    if layout_text != authority_text:
        reason = "layout_payload_mismatch"
    if len(runs) != len(span_masks) or not runs:
        evidence_failure("glyph_span_gap")

    union = np.zeros_like(core)
    occupied = np.zeros_like(core, dtype=bool)
    spans: list[RenderedGlyphSpanEvidence] = []
    cursor = 0
    for index, (run, span_mask) in enumerate(zip(runs, span_masks)):
        line = run.observed_text if isinstance(run, GlyphRunObservation) else ""
        if not isinstance(run, GlyphRunObservation) or not run.observed_text:
            evidence_failure("glyph_run_empty")
            continue
        if run.span_index != index or run.observed_text != line:
            evidence_failure("glyph_span_gap")
        start = authority_text.find(line, cursor)
        if start < 0:
            evidence_failure("glyph_span_gap")
            start = cursor
        end = min(len(authority_text), start + len(line))
        cursor = end
        positive = span_mask > 0
        if not np.any(positive):
            evidence_failure("glyph_run_empty")
            continue
        if np.any(occupied & positive):
            evidence_failure("glyph_span_duplicated")
        occupied |= positive
        union = np.maximum(union, span_mask)
        bbox = _mask_bbox(span_mask)
        if bbox is None:
            evidence_failure("glyph_run_empty")
            continue
        expected_glyphs = max(1, len([c for c in line if not c.isspace()]))
        observed_glyphs = len(run.clusters)
        spans.append(
            RenderedGlyphSpanEvidence(
                span_index=index,
                text_start=start,
                text_end=end,
                normalized_text_sha256=_sha_text(line),
                non_whitespace_glyph_count=observed_glyphs,
                glyph_coverage_ratio=min(1.0, observed_glyphs / float(expected_glyphs)),
                font_run_sha256=run.font_run_sha256,
                glyph_run_sha256=run.glyph_run_sha256,
                core_mask_sha256=_array_sha256(span_mask),
                core_mask_bbox_logical=bbox,
            )
        )
    if not np.array_equal(union, core):
        evidence_failure("glyph_mask_unbound")
    if reason == "delivered" and cursor != len(authority_text):
        reason = "glyph_span_gap"
    status = "delivered" if reason == "delivered" else "review_required"
    span_payload = [span.to_dict() for span in spans]
    spans_hash = _canonical_hash({"glyph_spans": span_payload})
    values = {
        "schema_version": 1,
        "owner_id": execution_authority.owner_id,
        "page_id": execution_authority.page_id,
        "execution_authority_sha256": execution_authority.authority_sha256,
        "source_payload_sha256": execution_authority.source_payload_sha256,
        "translated_payload_sha256": execution_authority.translated_payload_sha256,
        "layout_payload_sha256": _sha_text(layout_text),
        "rendered_payload_sha256": _sha_text(rendered_text),
        "rendered_patch_sha256": str(rendered_patch_sha256),
        "rendered_line_count": len(lines),
        "glyph_spans": tuple(spans),
        "glyph_spans_sha256": spans_hash,
        "glyph_core_mask_sha256": _array_sha256(core),
        "status": status,
        "reason": reason,
    }
    provisional = OwnerTextDeliveryContract(**values, contract_sha256="")
    return OwnerTextDeliveryContract(
        **values,
        contract_sha256=_canonical_hash(provisional._core_dict()),
    )


def validate_owner_text_delivery_contract(
    contract: OwnerTextDeliveryContract,
    *,
    execution_authority: OwnerTextExecutionAuthority,
) -> bool:
    if not isinstance(contract, OwnerTextDeliveryContract):
        raise TypeError("owner text delivery contract type mismatch")
    if _canonical_hash(contract._core_dict()) != contract.contract_sha256:
        raise ValueError("text delivery contract hash mismatch")
    if (
        contract.owner_id != execution_authority.owner_id
        or contract.page_id != execution_authority.page_id
        or contract.execution_authority_sha256 != execution_authority.authority_sha256
        or contract.source_payload_sha256 != execution_authority.source_payload_sha256
        or contract.translated_payload_sha256
        != execution_authority.translated_payload_sha256
    ):
        raise ValueError("translated execution authority mismatch")
    if contract.status != "delivered":
        raise ValueError(f"text delivery contract is not delivered:{contract.reason}")
    return True


def validate_owner_text_delivery_evidence(
    contract: OwnerTextDeliveryContract,
    *,
    execution_authority: OwnerTextExecutionAuthority,
    glyph_core_mask: np.ndarray,
    glyph_span_core_masks: Sequence[np.ndarray],
    glyph_span_runs: Sequence[GlyphRunObservation],
) -> bool:
    """Recompute the independent glyph proof carried by a committed patch."""

    validate_owner_text_delivery_contract(
        contract,
        execution_authority=execution_authority,
    )
    core = _canonical_mask(glyph_core_mask)
    if _array_sha256(core) != contract.glyph_core_mask_sha256:
        raise ValueError("glyph_core_mask_hash_mismatch")
    masks = tuple(
        _canonical_mask(mask, shape=core.shape) for mask in glyph_span_core_masks
    )
    runs = tuple(glyph_span_runs)
    spans = tuple(contract.glyph_spans)
    if not spans or len(spans) != len(masks) or len(spans) != len(runs):
        raise ValueError("glyph_span_gap")

    union = np.zeros_like(core)
    occupied = np.zeros_like(core, dtype=bool)
    rendered_parts: list[str] = []
    previous_end = 0
    for index, (span, mask, run) in enumerate(zip(spans, masks, runs)):
        if span.span_index != index or run.span_index != index:
            raise ValueError("glyph_span_gap")
        if _array_sha256(mask) != span.core_mask_sha256:
            raise ValueError("glyph_span_core_mask_hash_mismatch")
        if not run.observed_text or not np.any(mask):
            raise ValueError("glyph_run_empty")
        if _mask_bbox(mask) != span.core_mask_bbox_logical:
            raise ValueError("glyph_span_core_mask_bbox_mismatch")
        rebuilt = GlyphRunObservation.build(
            text=run.observed_text,
            font_identity=run.font_identity,
            span_index=index,
        )
        if rebuilt != run:
            raise ValueError("glyph_run_hash_mismatch")
        if (
            span.font_run_sha256 != run.font_run_sha256
            or span.glyph_run_sha256 != run.glyph_run_sha256
            or span.normalized_text_sha256 != _sha_text(run.observed_text)
            or span.non_whitespace_glyph_count != len(run.clusters)
            or span.glyph_coverage_ratio < 1.0
        ):
            raise ValueError("glyph_run_evidence_mismatch")
        if span.text_start < previous_end or span.text_end <= span.text_start:
            raise ValueError("glyph_span_gap")
        authorized_slice = normalize_delivery_payload(
            execution_authority.translated_payload[span.text_start:span.text_end]
        )
        if authorized_slice != run.observed_text:
            raise ValueError("translated_execution_authority_mismatch")
        previous_end = span.text_end
        positive = mask > 0
        if np.any(occupied & positive):
            raise ValueError("glyph_span_duplicated")
        occupied |= positive
        union = np.maximum(union, mask)
        rendered_parts.append(run.observed_text)

    if not np.array_equal(union, core):
        raise ValueError("glyph_mask_unbound")
    rendered = normalize_delivery_payload(" ".join(rendered_parts))
    if rendered != execution_authority.translated_payload:
        raise ValueError("translated_execution_authority_mismatch")
    if _sha_text(rendered) != contract.rendered_payload_sha256:
        raise ValueError("rendered_payload_hash_mismatch")
    if _canonical_hash({"glyph_spans": [span.to_dict() for span in spans]}) != contract.glyph_spans_sha256:
        raise ValueError("glyph_spans_hash_mismatch")
    return True
