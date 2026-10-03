"""Immutable page-coverage ledger and hash-bound recovery contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
import json
import logging
import os
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from .ocr_contract import (
    OCRAttempt,
    OCRBlock,
    OCRDiagnostics,
    OCRInvocationResult,
    OCRObservationRecord,
    OCRRequest,
    OCRTransformOperation,
    OCRTransformSpec,
    normalize_ocr_payload_text,
)

if TYPE_CHECKING:
    import numpy as np
    from .model import SourceTextComponent, TextObservation

from .hash_contract import canonical_json_bytes, canonical_json_sha256, sha256_bytes


logger = logging.getLogger(__name__)


BBox = tuple[int, int, int, int]
Point = tuple[int, int]
Polygon = tuple[Point, ...]
CoverageState = Literal[
    "discovered",
    "challenged",
    "observed",
    "owned",
    "target_ready",
    "execution_attempt",
    "repair_pending",
    "cleaned",
    "rendered",
    "final_verified",
    "explicit_non_dialogue_preserve",
    "review_required",
]
CANONICAL_COVERAGE_STATES = frozenset(CoverageState.__args__)
TERMINAL_COVERAGE_STATES = frozenset(
    {"final_verified", "explicit_non_dialogue_preserve", "review_required"}
)
RECOVERY_STATUSES = frozenset({"scheduled", "succeeded", "failed", "exhausted"})


class CoverageInvariantError(ValueError):
    """Raised when coverage cannot prove a complete page."""


class CoverageRecoveryExhausted(CoverageInvariantError):
    """Raised when a recovery chain reaches an exhausted decision."""


class CoverageIdentityError(CoverageInvariantError):
    """Raised when evidence crosses a page/run/execution identity boundary."""


def _jsonable(value: object) -> object:
    """Convert immutable contract values to canonical-JSON containers."""

    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _require_identity(*values: object) -> None:
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise CoverageInvariantError("coverage identity is incomplete")


def _require_sha256(value: str, label: str) -> None:
    if len(str(value or "")) != 64 or any(
        char not in "0123456789abcdef" for char in str(value).lower()
    ):
        raise CoverageInvariantError(f"{label} is not a canonical SHA-256")


def _canonical_polygon(value: Polygon | Sequence[Sequence[int]] | None) -> Polygon | None:
    if value is None:
        return None
    try:
        polygon = tuple((int(point[0]), int(point[1])) for point in value)
    except (IndexError, TypeError, ValueError) as exc:
        raise CoverageInvariantError("anchor polygon is invalid") from exc
    if len(polygon) < 3 or any(x < 0 or y < 0 for x, y in polygon):
        raise CoverageInvariantError("anchor polygon is invalid")
    return polygon


def _anchor_payload(
    *,
    component_id: str | None,
    observation_ids: Sequence[str],
    anchor_polygon_page: Polygon | None,
) -> dict[str, object]:
    observations = tuple(sorted({str(value) for value in observation_ids if str(value)}))
    polygon = _canonical_polygon(anchor_polygon_page)
    if component_id is not None:
        if not str(component_id).strip() or observations or polygon is not None:
            raise CoverageInvariantError(
                "recovery anchor must use component_id alone or an unassociated observation region"
            )
        return {"component_id": str(component_id)}
    if not observations or polygon is None:
        raise CoverageInvariantError(
            "unassociated recovery anchor requires observations and page-space polygon"
        )
    return {
        "observation_ids": list(observations),
        "polygon_page": [list(point) for point in polygon],
    }


def _attempt_fingerprint_payload(
    *,
    run_id: str,
    origin_execution_id: str,
    page_id: str,
    page_source_sha256: str,
    anchor_sha256: str,
    attempt_kind: str,
    transform_spec_sha256: str,
    geometry_sha256: str,
    input_pixel_sha256: str,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "origin_execution_id": origin_execution_id,
        "page_id": page_id,
        "page_source_sha256": page_source_sha256,
        "anchor_sha256": anchor_sha256,
        "attempt_kind": attempt_kind,
        "transform_spec_sha256": transform_spec_sha256,
        "geometry_sha256": geometry_sha256,
        "input_pixel_sha256": input_pixel_sha256,
    }


@dataclass(frozen=True)
class CoverageRecoveryRequest:
    request_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    component_id: str | None
    observation_ids: tuple[str, ...]
    anchor_polygon_page: Polygon | None
    anchor_sha256: str
    attempt_kind: str
    transform_spec_sha256: str
    geometry_sha256: str
    input_pixel_sha256: str
    attempt_fingerprint: str
    reason: str
    evidence_ids: tuple[str, ...]
    parent_decision_id: str | None
    parent_decision_sha256: str | None
    next_strategy: str | None
    request_sha256: str

    @classmethod
    def build(cls, **values: object) -> "CoverageRecoveryRequest":
        return build_recovery_request(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        _require_identity(
            self.request_id,
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.attempt_kind,
            self.reason,
        )
        for label, value in (
            ("page_source_sha256", self.page_source_sha256),
            ("transform_spec_sha256", self.transform_spec_sha256),
            ("geometry_sha256", self.geometry_sha256),
            ("input_pixel_sha256", self.input_pixel_sha256),
        ):
            _require_sha256(value, label)
        anchor = _anchor_payload(
            component_id=self.component_id,
            observation_ids=self.observation_ids,
            anchor_polygon_page=self.anchor_polygon_page,
        )
        if self.anchor_sha256 != canonical_json_sha256(anchor):
            raise CoverageInvariantError("recovery request anchor hash mismatch")
        fingerprint = canonical_json_sha256(
            _attempt_fingerprint_payload(
                run_id=self.run_id,
                origin_execution_id=self.origin_execution_id,
                page_id=self.page_id,
                page_source_sha256=self.page_source_sha256,
                anchor_sha256=self.anchor_sha256,
                attempt_kind=self.attempt_kind,
                transform_spec_sha256=self.transform_spec_sha256,
                geometry_sha256=self.geometry_sha256,
                input_pixel_sha256=self.input_pixel_sha256,
            )
        )
        if self.attempt_fingerprint != fingerprint:
            raise CoverageInvariantError("recovery request attempt fingerprint mismatch")
        if (self.parent_decision_id is None) != (self.parent_decision_sha256 is None):
            raise CoverageInvariantError("recovery request parent identity/hash must be paired")
        if self.request_sha256 != canonical_json_sha256(_request_payload(self)):
            raise CoverageInvariantError("recovery request hash mismatch")


def _request_payload(request: CoverageRecoveryRequest) -> dict[str, object]:
    payload = asdict(request)
    payload.pop("request_sha256", None)
    return _jsonable(payload)  # type: ignore[return-value]


@dataclass(frozen=True)
class CoverageRecoveryDecision:
    decision_id: str
    request_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    request_sha256: str
    component_id: str | None
    observation_ids: tuple[str, ...]
    anchor_polygon_page: Polygon | None
    anchor_sha256: str
    materialized_component_id: str | None
    attempt_kind: str
    transform_spec_sha256: str
    geometry_sha256: str
    input_pixel_sha256: str
    attempt_fingerprint: str
    attempt_id: str | None
    status: Literal["scheduled", "succeeded", "failed", "exhausted"]
    reason: str
    evidence_ids: tuple[str, ...]
    next_strategy: str | None
    decision_sha256: str

    @classmethod
    def build(cls, **values: object) -> "CoverageRecoveryDecision":
        return build_recovery_decision(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        _require_identity(
            self.decision_id,
            self.request_id,
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.reason,
        )
        if self.status not in RECOVERY_STATUSES:
            raise CoverageInvariantError("recovery decision status is invalid")
        if self.status == "failed" and not self.next_strategy:
            raise CoverageInvariantError("failed recovery requires next_strategy")
        if self.status in {"succeeded", "exhausted"} and self.next_strategy is not None:
            raise CoverageInvariantError(f"{self.status} recovery cannot declare next_strategy")
        if self.status != "succeeded" and self.materialized_component_id is not None:
            raise CoverageInvariantError("only succeeded recovery may materialize a component")
        anchor = _anchor_payload(
            component_id=self.component_id,
            observation_ids=self.observation_ids,
            anchor_polygon_page=self.anchor_polygon_page,
        )
        if self.anchor_sha256 != canonical_json_sha256(anchor):
            raise CoverageInvariantError("recovery decision anchor hash mismatch")
        if self.decision_sha256 != canonical_json_sha256(_decision_payload(self)):
            raise CoverageInvariantError("recovery decision hash mismatch")


def _decision_payload(decision: CoverageRecoveryDecision) -> dict[str, object]:
    payload = asdict(decision)
    payload.pop("decision_sha256", None)
    return _jsonable(payload)  # type: ignore[return-value]


@dataclass(frozen=True)
class CoverageComponentInventoryEntry:
    component_id: str
    origin: Literal["discovery", "recovery_materialization"]
    introduced_by_decision_id: str | None
    anchor_sha256: str
    ordinal: int
    inventory_entry_sha256: str

    def __post_init__(self) -> None:
        _require_identity(self.component_id, self.origin)
        _require_sha256(self.anchor_sha256, "inventory anchor_sha256")
        if self.origin not in {"discovery", "recovery_materialization"}:
            raise CoverageInvariantError("inventory origin is invalid")
        if self.origin == "discovery" and self.introduced_by_decision_id is not None:
            raise CoverageInvariantError("discovery inventory cannot reference recovery")
        if self.origin == "recovery_materialization" and not self.introduced_by_decision_id:
            raise CoverageInvariantError("recovery inventory requires decision identity")
        if self.ordinal < 0:
            raise CoverageInvariantError("inventory ordinal is invalid")
        if self.inventory_entry_sha256 != canonical_json_sha256(
            _inventory_entry_payload(self)
        ):
            raise CoverageInvariantError("inventory entry hash mismatch")


def _inventory_entry_payload(
    entry: CoverageComponentInventoryEntry,
) -> dict[str, object]:
    payload = asdict(entry)
    payload.pop("inventory_entry_sha256", None)
    return payload


@dataclass(frozen=True)
class CoverageObservationDisposition:
    observation_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    disposition: str
    reason: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CoverageEntry:
    component_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    bbox_page: BBox
    polygon_page: Polygon
    materiality: Literal["material", "non_text", "uncertain"]
    container_id: str | None = None
    ocr_attempt_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    owner_id: str | None = None
    semantic_role: str | None = None
    protection_conflict: bool = False
    protection_evidence_ids: tuple[str, ...] = ()
    state: CoverageState = "discovered"
    preserve_policy: str | None = None


@dataclass(frozen=True)
class PageCoverageLedger:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    inventory_version: int
    parent_ledger_sha256: str | None
    component_inventory: tuple[CoverageComponentInventoryEntry, ...]
    expected_observation_ids: tuple[str, ...]
    entries: tuple[CoverageEntry, ...]
    observation_dispositions: tuple[CoverageObservationDisposition, ...]
    recovery_requests: tuple[CoverageRecoveryRequest, ...]
    recovery_decisions: tuple[CoverageRecoveryDecision, ...]
    canonical_json_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        _require_identity(
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
        )
        _require_sha256(self.page_source_sha256, "page_source_sha256")
        if self.inventory_version < 1:
            raise CoverageInvariantError("inventory_version must be positive")
        payload_bytes = canonical_json_bytes(_jsonable(_ledger_payload(self)))
        if self.canonical_json_bytes != payload_bytes:
            raise CoverageInvariantError("coverage ledger canonical bytes mismatch")
        if self.sha256 != sha256_bytes(payload_bytes):
            raise CoverageInvariantError("coverage ledger hash mismatch")

    @property
    def expected_component_ids(self) -> tuple[str, ...]:
        return tuple(item.component_id for item in self.component_inventory)

    def entry(self, component_id: str) -> CoverageEntry:
        matches = tuple(item for item in self.entries if item.component_id == component_id)
        if len(matches) != 1:
            raise CoverageInvariantError(
                f"coverage entry identity is not unique: {component_id}"
            )
        return matches[0]

    def entry_for_bbox(self, bbox: BBox) -> CoverageEntry:
        matches = tuple(item for item in self.entries if item.bbox_page == tuple(bbox))
        if len(matches) != 1:
            raise CoverageInvariantError(f"coverage bbox identity is not unique: {bbox}")
        return matches[0]

    def require_complete(self) -> None:
        _validate_inventory_snapshot(self)
        expected_identity = (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
        )
        entries_by_component: dict[str, list[CoverageEntry]] = {}
        for entry in self.entries:
            entries_by_component.setdefault(entry.component_id, []).append(entry)
            actual_identity = (
                entry.run_id,
                entry.origin_execution_id,
                entry.page_id,
                entry.page_source_sha256,
            )
            if actual_identity != expected_identity:
                raise CoverageInvariantError(
                    f"coverage entry identity mismatch: {entry.component_id}"
                )
        if set(entries_by_component) != set(self.expected_component_ids):
            missing = sorted(set(self.expected_component_ids) - set(entries_by_component))
            extra = sorted(set(entries_by_component) - set(self.expected_component_ids))
            raise CoverageInvariantError(
                "coverage component inventory mismatch: " + ", ".join((*missing, *extra))
            )
        duplicated = sorted(
            component_id
            for component_id, values in entries_by_component.items()
            if len(values) != 1
        )
        if duplicated:
            raise CoverageInvariantError(
                "coverage component has multiple lifecycles: " + ", ".join(duplicated)
            )
        for entry in self.entries:
            _validate_complete_entry(entry)

        observation_counts: dict[str, int] = {}
        for entry in self.entries:
            for observation_id in entry.observation_ids:
                observation_counts[observation_id] = observation_counts.get(observation_id, 0) + 1
        for disposition in self.observation_dispositions:
            if (
                disposition.run_id,
                disposition.origin_execution_id,
                disposition.page_id,
                disposition.page_source_sha256,
            ) != expected_identity:
                raise CoverageInvariantError(
                    f"observation disposition identity mismatch: {disposition.observation_id}"
                )
            if not disposition.reason or not disposition.evidence_ids:
                raise CoverageInvariantError(
                    f"observation disposition is not auditable: {disposition.observation_id}"
                )
            observation_counts[disposition.observation_id] = (
                observation_counts.get(disposition.observation_id, 0) + 1
            )
        expected_observations = set(self.expected_observation_ids)
        if set(observation_counts) != expected_observations:
            unresolved = sorted(expected_observations.symmetric_difference(observation_counts))
            raise CoverageInvariantError(
                "coverage observation inventory mismatch: " + ", ".join(unresolved)
            )
        invalid_counts = sorted(
            observation_id
            for observation_id, count in observation_counts.items()
            if count != 1
        )
        if invalid_counts:
            raise CoverageInvariantError(
                "coverage observation must resolve exactly once: "
                + ", ".join(invalid_counts)
            )
        if self.recovery_requests or self.recovery_decisions:
            validate_recovery_chain(self.recovery_requests, self.recovery_decisions)
            if any(decision.status == "scheduled" for decision in self.recovery_decisions):
                raise CoverageInvariantError("coverage recovery remains scheduled")

    def extend_component_inventory(
        self,
        entry: CoverageComponentInventoryEntry,
        *,
        request: CoverageRecoveryRequest,
        decision: CoverageRecoveryDecision,
    ) -> "PageCoverageLedger":
        _validate_decision_against_request(request, decision)
        if decision.status != "succeeded":
            raise CoverageInvariantError("inventory append requires succeeded recovery")
        if decision.materialized_component_id != entry.component_id:
            raise CoverageInvariantError("inventory append materialized a different component")
        if entry.introduced_by_decision_id != decision.decision_id:
            raise CoverageInvariantError("inventory append is not bound to its decision")
        if entry.anchor_sha256 != request.anchor_sha256:
            raise CoverageInvariantError("inventory append changed recovery anchor")
        successor = build_page_coverage_ledger(
            run_id=self.run_id,
            origin_execution_id=self.origin_execution_id,
            page_id=self.page_id,
            page_source_sha256=self.page_source_sha256,
            inventory_version=self.inventory_version + 1,
            parent_ledger_sha256=self.sha256,
            component_inventory=(*self.component_inventory, entry),
            expected_observation_ids=self.expected_observation_ids,
            entries=self.entries,
            observation_dispositions=self.observation_dispositions,
            recovery_requests=(*self.recovery_requests, request),
            recovery_decisions=(*self.recovery_decisions, decision),
        )
        validate_inventory_successor(self, successor)
        return successor


def _validate_complete_entry(entry: CoverageEntry) -> None:
    if entry.materiality not in {"material", "non_text", "uncertain"}:
        raise CoverageInvariantError(f"component materiality invalid: {entry.component_id}")
    if not entry.ocr_attempt_ids:
        raise CoverageInvariantError(f"component lacks OCR attempt: {entry.component_id}")
    if entry.state not in TERMINAL_COVERAGE_STATES:
        raise CoverageInvariantError(
            f"component lifecycle is not terminal: {entry.component_id}/{entry.state}"
        )
    dialogue = bool(entry.semantic_role and "dialogue" in entry.semantic_role)
    if dialogue and (not entry.owner_id or entry.state != "final_verified"):
        raise CoverageInvariantError(
            f"dialogue component did not finish verified: {entry.component_id}"
        )
    if entry.state == "explicit_non_dialogue_preserve":
        if dialogue or not entry.preserve_policy or not entry.preserve_policy.startswith("policy:"):
            raise CoverageInvariantError(
                f"preserved component lacks explicit non-dialogue policy: {entry.component_id}"
            )
    elif entry.materiality == "non_text":
        raise CoverageInvariantError(
            f"non-text component lacks explicit preserve disposition: {entry.component_id}"
        )
    elif entry.materiality == "uncertain" and (
        entry.state != "review_required" or entry.owner_id is not None
        or entry.observation_ids or entry.preserve_policy
    ):
        raise CoverageInvariantError(
            f"uncertain component lacks review disposition: {entry.component_id}"
        )
    elif entry.materiality == "material" and entry.state == "review_required":
        raise CoverageInvariantError(
            f"material component cannot bypass ownership as review: {entry.component_id}"
        )


def _ledger_payload(ledger: PageCoverageLedger) -> dict[str, object]:
    return _jsonable({
        "run_id": ledger.run_id,
        "origin_execution_id": ledger.origin_execution_id,
        "page_id": ledger.page_id,
        "page_source_sha256": ledger.page_source_sha256,
        "inventory_version": ledger.inventory_version,
        "parent_ledger_sha256": ledger.parent_ledger_sha256,
        "component_inventory": [asdict(item) for item in ledger.component_inventory],
        "expected_observation_ids": list(ledger.expected_observation_ids),
        "entries": [asdict(item) for item in ledger.entries],
        "observation_dispositions": [
            asdict(item) for item in ledger.observation_dispositions
        ],
        "recovery_requests": [asdict(item) for item in ledger.recovery_requests],
        "recovery_decisions": [asdict(item) for item in ledger.recovery_decisions],
    })  # type: ignore[return-value]


def _validate_inventory_snapshot(ledger: PageCoverageLedger) -> None:
    ids = ledger.expected_component_ids
    if len(ids) != len(set(ids)):
        raise CoverageInvariantError("component inventory contains duplicate identity")
    if tuple(item.ordinal for item in ledger.component_inventory) != tuple(range(len(ids))):
        raise CoverageInvariantError("component inventory order/ordinal is not canonical")
    for entry in ledger.component_inventory:
        entry.__post_init__()
    if ledger.inventory_version == 1:
        if ledger.parent_ledger_sha256 is not None:
            raise CoverageInvariantError("initial inventory cannot have parent ledger")
        if any(item.origin != "discovery" for item in ledger.component_inventory):
            raise CoverageInvariantError("initial inventory must contain discovery entries only")


def build_recovery_request(
    *,
    request_id: str,
    run_id: str,
    origin_execution_id: str,
    page_id: str,
    page_source_sha256: str,
    component_id: str | None,
    observation_ids: Sequence[str],
    anchor_polygon_page: Polygon | None,
    attempt_kind: str,
    transform_spec_sha256: str,
    geometry_sha256: str,
    input_pixel_sha256: str,
    reason: str,
    evidence_ids: Sequence[str],
    parent_decision: CoverageRecoveryDecision | None,
    next_strategy: str | None,
) -> CoverageRecoveryRequest:
    observations = tuple(sorted({str(value) for value in observation_ids if str(value)}))
    polygon = _canonical_polygon(anchor_polygon_page)
    anchor_sha256 = canonical_json_sha256(
        _anchor_payload(
            component_id=component_id,
            observation_ids=observations,
            anchor_polygon_page=polygon,
        )
    )
    if parent_decision is not None:
        if parent_decision.status != "failed":
            raise CoverageInvariantError("recovery successor requires failed parent decision")
        if parent_decision.next_strategy != attempt_kind:
            raise CoverageInvariantError("recovery successor skipped parent strategy")
        expected = (
            parent_decision.run_id,
            parent_decision.origin_execution_id,
            parent_decision.page_id,
            parent_decision.page_source_sha256,
            parent_decision.anchor_sha256,
        )
        actual = (run_id, origin_execution_id, page_id, page_source_sha256, anchor_sha256)
        if actual != expected:
            raise CoverageInvariantError("recovery successor changed identity or anchor")
    fingerprint = canonical_json_sha256(
        _attempt_fingerprint_payload(
            run_id=run_id,
            origin_execution_id=origin_execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            anchor_sha256=anchor_sha256,
            attempt_kind=attempt_kind,
            transform_spec_sha256=transform_spec_sha256,
            geometry_sha256=geometry_sha256,
            input_pixel_sha256=input_pixel_sha256,
        )
    )
    values = {
        "request_id": request_id,
        "run_id": run_id,
        "origin_execution_id": origin_execution_id,
        "page_id": page_id,
        "page_source_sha256": page_source_sha256,
        "component_id": component_id,
        "observation_ids": observations,
        "anchor_polygon_page": polygon,
        "anchor_sha256": anchor_sha256,
        "attempt_kind": attempt_kind,
        "transform_spec_sha256": transform_spec_sha256,
        "geometry_sha256": geometry_sha256,
        "input_pixel_sha256": input_pixel_sha256,
        "attempt_fingerprint": fingerprint,
        "reason": reason,
        "evidence_ids": tuple(sorted({str(value) for value in evidence_ids if str(value)})),
        "parent_decision_id": parent_decision.decision_id if parent_decision else None,
        "parent_decision_sha256": parent_decision.decision_sha256 if parent_decision else None,
        "next_strategy": next_strategy,
    }
    return CoverageRecoveryRequest(
        **values,
        request_sha256=canonical_json_sha256(_jsonable(values)),
    )


def _validate_decision_against_request(
    request: CoverageRecoveryRequest,
    decision: CoverageRecoveryDecision,
) -> None:
    expected = (
        request.request_id,
        request.run_id,
        request.origin_execution_id,
        request.page_id,
        request.page_source_sha256,
        request.request_sha256,
        request.component_id,
        request.observation_ids,
        request.anchor_polygon_page,
        request.anchor_sha256,
        request.attempt_kind,
        request.transform_spec_sha256,
        request.geometry_sha256,
        request.input_pixel_sha256,
        request.attempt_fingerprint,
    )
    actual = (
        decision.request_id,
        decision.run_id,
        decision.origin_execution_id,
        decision.page_id,
        decision.page_source_sha256,
        decision.request_sha256,
        decision.component_id,
        decision.observation_ids,
        decision.anchor_polygon_page,
        decision.anchor_sha256,
        decision.attempt_kind,
        decision.transform_spec_sha256,
        decision.geometry_sha256,
        decision.input_pixel_sha256,
        decision.attempt_fingerprint,
    )
    if actual != expected:
        raise CoverageInvariantError("recovery decision does not bind its exact request")


def build_recovery_decision(
    *,
    decision_id: str,
    request: CoverageRecoveryRequest,
    materialized_component_id: str | None,
    attempt_id: str | None,
    status: Literal["scheduled", "succeeded", "failed", "exhausted"],
    reason: str,
    evidence_ids: Sequence[str],
    next_strategy: str | None,
) -> CoverageRecoveryDecision:
    values = {
        "decision_id": decision_id,
        "request_id": request.request_id,
        "run_id": request.run_id,
        "origin_execution_id": request.origin_execution_id,
        "page_id": request.page_id,
        "page_source_sha256": request.page_source_sha256,
        "request_sha256": request.request_sha256,
        "component_id": request.component_id,
        "observation_ids": request.observation_ids,
        "anchor_polygon_page": request.anchor_polygon_page,
        "anchor_sha256": request.anchor_sha256,
        "materialized_component_id": materialized_component_id,
        "attempt_kind": request.attempt_kind,
        "transform_spec_sha256": request.transform_spec_sha256,
        "geometry_sha256": request.geometry_sha256,
        "input_pixel_sha256": request.input_pixel_sha256,
        "attempt_fingerprint": request.attempt_fingerprint,
        "attempt_id": attempt_id,
        "status": status,
        "reason": reason,
        "evidence_ids": tuple(sorted({str(value) for value in evidence_ids if str(value)})),
        "next_strategy": next_strategy,
    }
    decision = CoverageRecoveryDecision(
        **values,
        decision_sha256=canonical_json_sha256(_jsonable(values)),
    )
    _validate_decision_against_request(request, decision)
    return decision


def build_component_inventory_entry(
    *,
    component_id: str,
    origin: Literal["discovery", "recovery_materialization"],
    introduced_by_decision_id: str | None,
    anchor_polygon_page: Polygon,
    ordinal: int,
    decision: CoverageRecoveryDecision | None = None,
) -> CoverageComponentInventoryEntry:
    polygon = _canonical_polygon(anchor_polygon_page)
    assert polygon is not None
    anchor_sha256 = canonical_json_sha256(
        {"component_id": component_id, "polygon_page": [list(point) for point in polygon]}
    )
    if origin == "recovery_materialization":
        if decision is None or decision.status != "succeeded":
            raise CoverageInvariantError("recovery inventory entry requires succeeded decision")
        if decision.materialized_component_id != component_id:
            raise CoverageInvariantError("inventory entry component differs from decision")
        if decision.decision_id != introduced_by_decision_id:
            raise CoverageInvariantError("inventory entry decision identity mismatch")
        if decision.anchor_polygon_page is not None and decision.anchor_polygon_page != polygon:
            raise CoverageInvariantError("inventory entry polygon differs from recovery")
        anchor_sha256 = decision.anchor_sha256
    values = {
        "component_id": component_id,
        "origin": origin,
        "introduced_by_decision_id": introduced_by_decision_id,
        "anchor_sha256": anchor_sha256,
        "ordinal": ordinal,
    }
    return CoverageComponentInventoryEntry(
        **values,
        inventory_entry_sha256=canonical_json_sha256(values),
    )


def build_page_coverage_ledger(
    *,
    run_id: str,
    origin_execution_id: str,
    page_id: str,
    page_source_sha256: str,
    inventory_version: int,
    parent_ledger_sha256: str | None,
    component_inventory: Sequence[CoverageComponentInventoryEntry],
    expected_observation_ids: Sequence[str],
    entries: Sequence[CoverageEntry],
    observation_dispositions: Sequence[CoverageObservationDisposition] = (),
    recovery_requests: Sequence[CoverageRecoveryRequest] = (),
    recovery_decisions: Sequence[CoverageRecoveryDecision] = (),
) -> PageCoverageLedger:
    values = {
        "run_id": run_id,
        "origin_execution_id": origin_execution_id,
        "page_id": page_id,
        "page_source_sha256": page_source_sha256,
        "inventory_version": inventory_version,
        "parent_ledger_sha256": parent_ledger_sha256,
        "component_inventory": tuple(component_inventory),
        "expected_observation_ids": tuple(
            sorted({str(value) for value in expected_observation_ids if str(value)})
        ),
        "entries": tuple(entries),
        "observation_dispositions": tuple(observation_dispositions),
        "recovery_requests": tuple(recovery_requests),
        "recovery_decisions": tuple(recovery_decisions),
    }
    payload = {
        "run_id": run_id,
        "origin_execution_id": origin_execution_id,
        "page_id": page_id,
        "page_source_sha256": page_source_sha256,
        "inventory_version": inventory_version,
        "parent_ledger_sha256": parent_ledger_sha256,
        "component_inventory": [asdict(item) for item in values["component_inventory"]],
        "expected_observation_ids": list(values["expected_observation_ids"]),
        "entries": [asdict(item) for item in values["entries"]],
        "observation_dispositions": [
            asdict(item) for item in values["observation_dispositions"]
        ],
        "recovery_requests": [asdict(item) for item in values["recovery_requests"]],
        "recovery_decisions": [asdict(item) for item in values["recovery_decisions"]],
    }
    payload_bytes = canonical_json_bytes(_jsonable(payload))
    ledger = PageCoverageLedger(
        **values,
        canonical_json_bytes=payload_bytes,
        sha256=sha256_bytes(payload_bytes),
    )
    _validate_inventory_snapshot(ledger)
    return ledger


def validate_inventory_successor(
    previous: PageCoverageLedger,
    successor: PageCoverageLedger,
) -> None:
    identity_previous = (
        previous.run_id,
        previous.origin_execution_id,
        previous.page_id,
        previous.page_source_sha256,
    )
    identity_successor = (
        successor.run_id,
        successor.origin_execution_id,
        successor.page_id,
        successor.page_source_sha256,
    )
    if identity_previous != identity_successor:
        raise CoverageInvariantError("inventory successor changed page identity")
    if successor.inventory_version != previous.inventory_version + 1:
        raise CoverageInvariantError("inventory successor version is not monotonic")
    if successor.parent_ledger_sha256 != previous.sha256:
        raise CoverageInvariantError("inventory successor parent hash mismatch")
    prefix = successor.component_inventory[: len(previous.component_inventory)]
    if prefix != previous.component_inventory or len(successor.component_inventory) != len(prefix) + 1:
        raise CoverageInvariantError("inventory successor changed immutable prefix")
    appended = successor.component_inventory[-1]
    if appended.origin != "recovery_materialization" or appended.ordinal != len(prefix):
        raise CoverageInvariantError("inventory successor append is not recovery-bound")
    matching_decisions = tuple(
        decision
        for decision in successor.recovery_decisions
        if decision.decision_id == appended.introduced_by_decision_id
    )
    if len(matching_decisions) != 1:
        raise CoverageInvariantError("inventory successor lacks exactly one recovery decision")
    decision = matching_decisions[0]
    matching_requests = tuple(
        request
        for request in successor.recovery_requests
        if request.request_id == decision.request_id
    )
    if len(matching_requests) != 1:
        raise CoverageInvariantError("inventory successor lacks exact recovery request")
    request = matching_requests[0]
    _validate_decision_against_request(request, decision)
    if (
        decision.status != "succeeded"
        or decision.materialized_component_id != appended.component_id
        or decision.anchor_sha256 != appended.anchor_sha256
    ):
        raise CoverageInvariantError("inventory successor recovery proof is invalid")
    _validate_inventory_snapshot(successor)


def validate_recovery_chain(
    requests: Sequence[CoverageRecoveryRequest],
    decisions: Sequence[CoverageRecoveryDecision],
) -> None:
    request_by_id: dict[str, CoverageRecoveryRequest] = {}
    for request in requests:
        if request.request_id in request_by_id:
            raise CoverageInvariantError(f"duplicate recovery request: {request.request_id}")
        request.__post_init__()
        request_by_id[request.request_id] = request
    decision_by_id: dict[str, CoverageRecoveryDecision] = {}
    for decision in decisions:
        if decision.decision_id in decision_by_id:
            raise CoverageInvariantError(f"duplicate recovery decision: {decision.decision_id}")
        decision.__post_init__()
        request = request_by_id.get(decision.request_id)
        if request is None:
            raise CoverageInvariantError(f"orphan recovery decision: {decision.decision_id}")
        _validate_decision_against_request(request, decision)
        decision_by_id[decision.decision_id] = decision

    children_by_decision: dict[str, list[CoverageRecoveryRequest]] = {}
    for request in requests:
        if request.parent_decision_id is None:
            continue
        parent = decision_by_id.get(request.parent_decision_id)
        if parent is None:
            raise CoverageInvariantError(f"orphan recovery request: {request.request_id}")
        if request.parent_decision_sha256 != parent.decision_sha256:
            raise CoverageInvariantError("recovery request parent hash mismatch")
        if parent.status != "failed" or request.attempt_kind != parent.next_strategy:
            raise CoverageInvariantError("recovery request skipped parent strategy")
        parent_request = request_by_id[parent.request_id]
        if (
            request.run_id,
            request.origin_execution_id,
            request.page_id,
            request.page_source_sha256,
            request.anchor_sha256,
        ) != (
            parent_request.run_id,
            parent_request.origin_execution_id,
            parent_request.page_id,
            parent_request.page_source_sha256,
            parent_request.anchor_sha256,
        ):
            raise CoverageInvariantError("recovery request changed parent identity or anchor")
        children_by_decision.setdefault(parent.decision_id, []).append(request)

    for decision in decisions:
        children = children_by_decision.get(decision.decision_id, [])
        if decision.status == "failed" and len(children) != 1:
            raise CoverageInvariantError("failed recovery requires exactly one successor")
        if decision.status != "failed" and children:
            raise CoverageInvariantError("terminal recovery decision cannot have successor")
        if decision.status == "exhausted":
            raise CoverageRecoveryExhausted(
                f"coverage recovery exhausted: {decision.decision_id}"
            )


def _component_payload(component: "SourceTextComponent") -> dict[str, object]:
    return _jsonable(asdict(component))  # type: ignore[return-value]


def _observation_payload(observation: "TextObservation") -> dict[str, object]:
    return _jsonable(asdict(observation))  # type: ignore[return-value]


def _request_json(request: OCRRequest) -> dict[str, object]:
    return _jsonable(asdict(request))  # type: ignore[return-value]


def _record_json(record: OCRObservationRecord) -> dict[str, object]:
    return _jsonable(asdict(record))  # type: ignore[return-value]


def _invocation_json(invocation: OCRInvocationResult) -> dict[str, object]:
    return {
        "request": _request_json(invocation.request),
        "blocks": [
            {
                "block_id": item.block_id,
                "text": item.text,
                "confidence": item.confidence,
                "bbox_page": list(item.bbox_page),
                "polygon_page": [list(point) for point in item.polygon_page],
                "extras": _jsonable(item.extras),
            }
            for item in invocation.blocks
        ],
        "observations": [_record_json(item) for item in invocation.observations],
        "full_page_lines": [_record_json(item) for item in invocation.full_page_lines],
        "attempts": [item.to_json() for item in invocation.attempts],
        "attempt_chain_sha256": invocation.attempt_chain_sha256,
        "diagnostics": {
            "provider": invocation.diagnostics.provider,
            "extras": _jsonable(invocation.diagnostics.extras),
        },
    }


def _ledger_json(ledger: PageCoverageLedger) -> dict[str, object]:
    return {
        "payload": json.loads(ledger.canonical_json_bytes.decode("utf-8")),
        "sha256": ledger.sha256,
    }


def _coverage_result_payload(values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": values["run_id"],
        "origin_execution_id": values["origin_execution_id"],
        "page_id": values["page_id"],
        "page_source_sha256": values["page_source_sha256"],
        "ledger_history": [_ledger_json(item) for item in values["ledger_history"]],
        "components": [_component_payload(item) for item in values["components"]],
        "observations": [_observation_payload(item) for item in values["observations"]],
        "ocr_requests": [_request_json(item) for item in values["ocr_requests"]],
        "ocr_invocations": [_invocation_json(item) for item in values["ocr_invocations"]],
        "recovery_requests": [
            _jsonable(asdict(item)) for item in values["recovery_requests"]
        ],
        "recovery_decisions": [
            _jsonable(asdict(item)) for item in values["recovery_decisions"]
        ],
        "pending_request_ids": list(values["pending_request_ids"]),
    }


def _require_append_only(previous: Sequence[object], current: Sequence[object], label: str) -> None:
    if len(current) < len(previous) or tuple(current[: len(previous)]) != tuple(previous):
        raise CoverageInvariantError(f"{label} history is not append-only")


def _require_observation_history(
    previous: Sequence["TextObservation"], current: Sequence["TextObservation"]
) -> None:
    if len(current) < len(previous):
        raise CoverageInvariantError("observations history is not append-only")
    for old, new in zip(previous, current):
        if old == new:
            continue
        if replace(new, component_ids=old.component_ids) != old:
            raise CoverageInvariantError("observation history changed immutable evidence")
        if not set(old.component_ids) <= set(new.component_ids):
            raise CoverageInvariantError("observation association is not monotonic")


def _pending_recovery_request_ids(
    requests: Sequence[CoverageRecoveryRequest],
    decisions: Sequence[CoverageRecoveryDecision],
) -> tuple[str, ...]:
    terminal_by_request: dict[str, CoverageRecoveryDecision] = {}
    scheduled_by_request: set[str] = set()
    request_ids = {request.request_id for request in requests}
    for decision in decisions:
        if decision.request_id not in request_ids:
            raise CoverageInvariantError(f"orphan recovery decision: {decision.decision_id}")
        if decision.status == "scheduled":
            if decision.request_id in scheduled_by_request or decision.request_id in terminal_by_request:
                raise CoverageInvariantError("recovery scheduled decision is duplicated or late")
            scheduled_by_request.add(decision.request_id)
            continue
        if decision.request_id in terminal_by_request:
            raise CoverageInvariantError("recovery request has multiple terminal decisions")
        terminal_by_request[decision.request_id] = decision
    return tuple(
        request.request_id
        for request in requests
        if request.request_id not in terminal_by_request
    )


def _identity_from_values(values: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(values["run_id"]),
        str(values["origin_execution_id"]),
        str(values["page_id"]),
        str(values["page_source_sha256"]),
    )


@dataclass(frozen=True)
class PageCoverageResult:
    """Immutable page-global coverage snapshot with append-only evidence."""

    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    ledger_history: tuple[PageCoverageLedger, ...]
    components: tuple["SourceTextComponent", ...]
    observations: tuple["TextObservation", ...]
    ocr_requests: tuple[OCRRequest, ...]
    ocr_invocations: tuple[OCRInvocationResult, ...]
    recovery_requests: tuple[CoverageRecoveryRequest, ...]
    recovery_decisions: tuple[CoverageRecoveryDecision, ...]
    pending_request_ids: tuple[str, ...]
    canonical_json_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        values = self._values()
        encoded = canonical_json_bytes(_coverage_result_payload(values))
        if encoded != self.canonical_json_bytes:
            raise CoverageInvariantError("coverage result canonical bytes mismatch")
        if sha256_bytes(encoded) != self.sha256:
            raise CoverageInvariantError("coverage result hash mismatch")
        self._validate_snapshot()

    def _values(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in {"canonical_json_bytes", "sha256"}
        }

    @classmethod
    def _build(cls, **values: object) -> "PageCoverageResult":
        normalized = dict(values)
        normalized["ledger_history"] = tuple(normalized.get("ledger_history") or ())
        normalized["components"] = tuple(normalized.get("components") or ())
        normalized["observations"] = tuple(normalized.get("observations") or ())
        normalized["ocr_requests"] = tuple(normalized.get("ocr_requests") or ())
        normalized["ocr_invocations"] = tuple(normalized.get("ocr_invocations") or ())
        normalized["recovery_requests"] = tuple(normalized.get("recovery_requests") or ())
        normalized["recovery_decisions"] = tuple(normalized.get("recovery_decisions") or ())
        derived_pending = _pending_recovery_request_ids(
            normalized["recovery_requests"],
            normalized["recovery_decisions"],
        )
        supplied_pending = normalized.get("pending_request_ids")
        if supplied_pending is not None and tuple(supplied_pending) != derived_pending:
            raise CoverageInvariantError("pending recovery request view is inconsistent")
        normalized["pending_request_ids"] = derived_pending
        encoded = canonical_json_bytes(_coverage_result_payload(normalized))
        return cls(
            **normalized,
            canonical_json_bytes=encoded,
            sha256=sha256_bytes(encoded),
        )

    @classmethod
    def initialize(
        cls,
        *,
        run_id: str,
        origin_execution_id: str,
        page_id: str,
        page_source_sha256: str,
        components: tuple["SourceTextComponent", ...],
    ) -> "PageCoverageResult":
        inventory = tuple(
            build_component_inventory_entry(
                component_id=component.component_id,
                origin="discovery",
                introduced_by_decision_id=None,
                anchor_polygon_page=component.polygon_page,
                ordinal=index,
            )
            for index, component in enumerate(components)
        )
        entries = tuple(
            CoverageEntry(
                component_id=component.component_id,
                run_id=run_id,
                origin_execution_id=origin_execution_id,
                page_id=page_id,
                page_source_sha256=page_source_sha256,
                bbox_page=component.bbox_page,
                polygon_page=component.polygon_page,
                materiality="material",
            )
            for component in components
        )
        ledger = build_page_coverage_ledger(
            run_id=run_id,
            origin_execution_id=origin_execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            inventory_version=1,
            parent_ledger_sha256=None,
            component_inventory=inventory,
            expected_observation_ids=(),
            entries=entries,
        )
        return cls._build(
            run_id=run_id,
            origin_execution_id=origin_execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            ledger_history=(ledger,),
            components=components,
            observations=(),
            ocr_requests=(),
            ocr_invocations=(),
            recovery_requests=(),
            recovery_decisions=(),
        )

    @classmethod
    def build_from(cls, previous: "PageCoverageResult", **changes: object) -> "PageCoverageResult":
        values = previous._values()
        values.update(changes)
        if "pending_request_ids" not in changes:
            values["pending_request_ids"] = None
        if _identity_from_values(values) != (
            previous.run_id,
            previous.origin_execution_id,
            previous.page_id,
            previous.page_source_sha256,
        ):
            raise CoverageIdentityError("coverage successor changed page identity")
        if "ledger_history" not in changes and any(
            name in changes
            for name in ("observations", "recovery_requests", "recovery_decisions")
        ):
            observations = tuple(values["observations"])
            observation_by_component: dict[str, list[object]] = {}
            for observation in observations:
                for component_id in observation.component_ids:
                    observation_by_component.setdefault(component_id, []).append(observation)
            entries = tuple(
                replace(
                    entry,
                    ocr_attempt_ids=tuple(
                        dict.fromkeys(
                            (
                                *entry.ocr_attempt_ids,
                                *(
                                    item.attempt_id
                                    for item in observation_by_component.get(
                                        entry.component_id, ()
                                    )
                                ),
                            )
                        )
                    ),
                    observation_ids=tuple(
                        item.observation_id
                        for item in observation_by_component.get(entry.component_id, ())
                    ),
                    state=(
                        "observed"
                        if observation_by_component.get(entry.component_id)
                        and entry.state in {"discovered", "challenged", "observed"}
                        else entry.state
                    ),
                )
                for entry in previous.ledger.entries
            )
            successor = build_page_coverage_ledger(
                run_id=previous.run_id,
                origin_execution_id=previous.origin_execution_id,
                page_id=previous.page_id,
                page_source_sha256=previous.page_source_sha256,
                inventory_version=previous.ledger.inventory_version + 1,
                parent_ledger_sha256=previous.ledger.sha256,
                component_inventory=previous.ledger.component_inventory,
                expected_observation_ids=tuple(
                    item.observation_id for item in observations
                ),
                entries=entries,
                observation_dispositions=previous.ledger.observation_dispositions,
                recovery_requests=tuple(values["recovery_requests"]),
                recovery_decisions=tuple(values["recovery_decisions"]),
            )
            values["ledger_history"] = (*previous.ledger_history, successor)
        for name in (
            "ledger_history",
            "components",
            "ocr_requests",
            "ocr_invocations",
            "recovery_requests",
            "recovery_decisions",
        ):
            _require_append_only(getattr(previous, name), tuple(values[name]), name)
        _require_observation_history(
            previous.observations, tuple(values["observations"])
        )
        if len(tuple(values["ledger_history"])) > len(previous.ledger_history) + 1:
            raise CoverageInvariantError("coverage successor appended multiple ledger versions")
        return cls._build(**values)

    @property
    def ledger(self) -> PageCoverageLedger:
        if not self.ledger_history:
            raise CoverageInvariantError("coverage result has no ledger")
        return self.ledger_history[-1]

    @property
    def entries(self) -> tuple[CoverageEntry, ...]:
        return self.ledger.entries

    @property
    def pending_requests(self) -> tuple[CoverageRecoveryRequest, ...]:
        pending = set(self.pending_request_ids)
        return tuple(item for item in self.recovery_requests if item.request_id in pending)

    def entry(self, component_id: str) -> CoverageEntry:
        return self.ledger.entry(component_id)

    def entry_for_bbox(self, bbox: BBox) -> CoverageEntry:
        return self.ledger.entry_for_bbox(bbox)

    def _validate_snapshot(self) -> None:
        identity = (self.run_id, self.origin_execution_id, self.page_id, self.page_source_sha256)
        _require_identity(*identity)
        _require_sha256(self.page_source_sha256, "page_source_sha256")
        if not self.ledger_history:
            raise CoverageInvariantError("coverage result requires ledger history")
        for index, ledger in enumerate(self.ledger_history):
            if (ledger.run_id, ledger.origin_execution_id, ledger.page_id, ledger.page_source_sha256) != identity:
                raise CoverageIdentityError("coverage ledger belongs to another page execution")
            if index == 0:
                if ledger.parent_ledger_sha256 is not None:
                    raise CoverageInvariantError("initial coverage ledger has a parent")
            else:
                previous = self.ledger_history[index - 1]
                if ledger.parent_ledger_sha256 != previous.sha256:
                    raise CoverageInvariantError("coverage parent_ledger_sha256 chain is broken")
                if ledger.inventory_version != previous.inventory_version + 1:
                    raise CoverageInvariantError("coverage ledger version chain is not monotonic")
                if ledger.component_inventory != previous.component_inventory:
                    validate_inventory_successor(previous, ledger)
        component_ids = tuple(item.component_id for item in self.components)
        if component_ids != self.ledger.expected_component_ids or len(component_ids) != len(set(component_ids)):
            raise CoverageInvariantError("coverage components differ from ledger inventory")
        outer_observation_ids = tuple(item.observation_id for item in self.observations)
        if len(outer_observation_ids) != len(set(outer_observation_ids)) or tuple(
            sorted(outer_observation_ids)
        ) != tuple(self.ledger.expected_observation_ids):
            raise CoverageInvariantError("coverage observations differ from ledger inventory")
        if self.ledger.recovery_requests != self.recovery_requests:
            raise CoverageInvariantError("coverage recovery requests differ from ledger")
        if self.ledger.recovery_decisions != self.recovery_decisions:
            raise CoverageInvariantError("coverage recovery decisions differ from ledger")

        request_identities: set[tuple[str, ...]] = set()
        for request in self.ocr_requests:
            if request.identity[:4] != identity:
                raise CoverageIdentityError("OCR request belongs to another page execution")
            if request.identity in request_identities:
                raise CoverageInvariantError("duplicate OCR request identity")
            request_identities.add(request.identity)
        invocation_by_id: dict[str, OCRInvocationResult] = {}
        attempts_by_id: dict[str, OCRAttempt] = {}
        for invocation in self.ocr_invocations:
            if invocation.request.identity not in request_identities:
                raise CoverageIdentityError("OCR invocation has no preserved request")
            if invocation.invocation_id in invocation_by_id:
                raise CoverageInvariantError("duplicate OCR invocation identity")
            invocation_by_id[invocation.invocation_id] = invocation
            for attempt in invocation.attempts:
                if attempt.attempt_id in attempts_by_id:
                    raise CoverageInvariantError("duplicate OCR attempt identity")
                attempts_by_id[attempt.attempt_id] = attempt
        observation_ids: set[str] = set()
        for observation in self.observations:
            if (
                observation.run_id,
                observation.origin_execution_id,
                observation.page_id,
                observation.page_source_sha256,
            ) != identity:
                raise CoverageIdentityError("OCR observation belongs to another page execution")
            if observation.observation_id in observation_ids:
                raise CoverageInvariantError("duplicate coverage observation identity")
            observation_ids.add(observation.observation_id)
            attempt = attempts_by_id.get(observation.attempt_id)
            if attempt is None or attempt.invocation_id != observation.invocation_id:
                raise CoverageInvariantError("coverage observation has orphan OCR attempt")
            if observation.payload_sha256 != sha256_bytes(
                normalize_ocr_payload_text(observation.text).encode("utf-8")
            ):
                raise CoverageInvariantError("coverage observation payload hash mismatch")
        if tuple(item.request_id for item in self.recovery_requests) != tuple(
            dict.fromkeys(item.request_id for item in self.recovery_requests)
        ):
            raise CoverageInvariantError("duplicate recovery request identity")
        fingerprints = [item.attempt_fingerprint for item in self.recovery_requests]
        if len(fingerprints) != len(set(fingerprints)):
            raise CoverageInvariantError("duplicate recovery attempt_fingerprint")
        for item in (*self.recovery_requests, *self.recovery_decisions):
            if (item.run_id, item.origin_execution_id, item.page_id, item.page_source_sha256) != identity:
                raise CoverageIdentityError("recovery evidence belongs to another page execution")
        request_by_id = {item.request_id: item for item in self.recovery_requests}
        for decision in self.recovery_decisions:
            request = request_by_id.get(decision.request_id)
            if request is None:
                raise CoverageInvariantError(
                    f"orphan recovery decision: {decision.decision_id}"
                )
            _validate_decision_against_request(request, decision)
        if self.pending_request_ids != _pending_recovery_request_ids(
            self.recovery_requests, self.recovery_decisions
        ):
            raise CoverageInvariantError("pending recovery request view is inconsistent")

    def require_ready_for_ownership(self) -> None:
        self._validate_snapshot()
        if self.pending_request_ids:
            raise CoverageInvariantError("coverage has pending recovery requests")
        if self.recovery_requests or self.recovery_decisions:
            validate_recovery_chain(self.recovery_requests, self.recovery_decisions)
            terminal_by_request = {
                item.request_id: item
                for item in self.recovery_decisions
                if item.status in {"succeeded", "failed", "exhausted"}
            }
            if any(item.status != "succeeded" for item in terminal_by_request.values() if not any(
                child.parent_decision_id == item.decision_id for child in self.recovery_requests
            )):
                raise CoverageInvariantError("coverage recovery chain did not terminate in success")
        attempt_ids = {
            attempt.attempt_id
            for invocation in self.ocr_invocations
            for attempt in invocation.attempts
        }
        observation_ids = {item.observation_id for item in self.observations}
        component_ids = {item.component_id for item in self.components}
        for observation in self.observations:
            if not observation.component_ids:
                raise CoverageInvariantError(
                    f"observation remains unassociated: {observation.observation_id}"
                )
            if not set(observation.component_ids) <= component_ids:
                raise CoverageInvariantError(
                    f"observation references unknown component: {observation.observation_id}"
                )
        for entry in self.entries:
            if entry.materiality == "uncertain":
                if (
                    entry.state != "review_required"
                    or not entry.ocr_attempt_ids
                    or entry.observation_ids
                    or entry.owner_id is not None
                    or entry.preserve_policy is not None
                ):
                    raise CoverageInvariantError(
                        f"uncertain component lacks auditable review state: {entry.component_id}"
                    )
            if entry.materiality == "material" and not entry.ocr_attempt_ids:
                raise CoverageInvariantError(f"component lacks OCR attempt: {entry.component_id}")
            if not set(entry.ocr_attempt_ids) <= attempt_ids:
                raise CoverageInvariantError(f"component has orphan OCR attempt: {entry.component_id}")
            if not set(entry.observation_ids) <= observation_ids:
                raise CoverageInvariantError(f"component has orphan observation: {entry.component_id}")
            if entry.state == "explicit_non_dialogue_preserve":
                if (
                    not entry.preserve_policy
                    or not entry.preserve_policy.startswith("policy:")
                    or bool(entry.semantic_role and "dialogue" in entry.semantic_role)
                ):
                    raise CoverageInvariantError(
                        f"preserved component lacks explicit non-dialogue policy: {entry.component_id}"
                    )
                continue
            if entry.materiality == "material" and not entry.observation_ids:
                raise CoverageInvariantError(
                    f"translatable component lacks OCR observation: {entry.component_id}"
                )
            if entry.materiality == "material" and not entry.container_id:
                raise CoverageInvariantError(
                    f"translatable component lacks container: {entry.component_id}"
                )

    @classmethod
    def from_canonical_json_bytes(cls, encoded: bytes) -> "PageCoverageResult":
        if not isinstance(encoded, bytes):
            raise TypeError("coverage canonical payload must be bytes")
        payload = json.loads(encoded.decode("utf-8"))
        if payload.get("schema_version") != 1:
            raise CoverageInvariantError("unsupported coverage result schema")
        ledgers = tuple(_ledger_from_json(item) for item in payload["ledger_history"])
        components = tuple(_component_from_json(item) for item in payload["components"])
        observations = tuple(_text_observation_from_json(item) for item in payload["observations"])
        requests = tuple(OCRRequest(**item) for item in payload["ocr_requests"])
        invocations = tuple(_invocation_from_json(item) for item in payload["ocr_invocations"])
        recovery_requests = tuple(_recovery_request_from_json(item) for item in payload["recovery_requests"])
        recovery_decisions = tuple(_recovery_decision_from_json(item) for item in payload["recovery_decisions"])
        result = cls._build(
            run_id=payload["run_id"],
            origin_execution_id=payload["origin_execution_id"],
            page_id=payload["page_id"],
            page_source_sha256=payload["page_source_sha256"],
            ledger_history=ledgers,
            components=components,
            observations=observations,
            ocr_requests=requests,
            ocr_invocations=invocations,
            recovery_requests=recovery_requests,
            recovery_decisions=recovery_decisions,
            pending_request_ids=tuple(payload["pending_request_ids"]),
        )
        if result.canonical_json_bytes != encoded:
            raise CoverageInvariantError("coverage canonical reopen changed bytes")
        return result


def _component_from_json(payload: Mapping[str, object]) -> "SourceTextComponent":
    from .model import SourceTextComponent

    values = dict(payload)
    values["bbox_page"] = tuple(values["bbox_page"])
    values["polygon_page"] = tuple(tuple(point) for point in values["polygon_page"])
    for name in ("detector_sources", "script_evidence", "evidence_ids"):
        values[name] = tuple(values.get(name) or ())
    return SourceTextComponent(**values)


def _text_observation_from_json(payload: Mapping[str, object]) -> "TextObservation":
    from .model import TextObservation

    values = dict(payload)
    for name in ("bbox_page", "source_bbox_page", "text_pixel_bbox_page", "layout_bbox_page"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    values["component_ids"] = tuple(values.get("component_ids") or ())
    values["polygons_page"] = tuple(
        tuple(tuple(point) for point in polygon)
        for polygon in values.get("polygons_page") or ()
    )
    for name in ("tile_provenance", "projection_ids", "line_texts"):
        values[name] = tuple(values.get(name) or ())
    return TextObservation(**values)


def _transform_from_attempt_json(payload: Mapping[str, object]) -> OCRTransformSpec:
    raw = json.loads(str(payload["transform_spec_canonical_json"]))
    operations = []
    for item in raw["operations"]:
        values = dict(item)
        for name in ("bbox_page", "output_size", "border_value_rgb", "affine_matrix_fixed_1e6"):
            if values.get(name) is not None:
                values[name] = tuple(values[name])
        operations.append(OCRTransformOperation(**values))
    spec = OCRTransformSpec.build(tuple(operations))
    if spec.sha256 != payload["transform_spec_sha256"]:
        raise CoverageInvariantError("OCR transform hash changed during reopen")
    return spec


def _record_from_json(payload: Mapping[str, object]) -> OCRObservationRecord:
    values = dict(payload)
    values["bbox_page"] = tuple(values["bbox_page"])
    values["polygon_page"] = tuple(tuple(point) for point in values["polygon_page"])
    return OCRObservationRecord(**values)


def _invocation_from_json(payload: Mapping[str, object]) -> OCRInvocationResult:
    request = OCRRequest(**payload["request"])
    attempts = []
    for item in payload["attempts"]:
        values = dict(item)
        values.pop("transform_spec_canonical_json")
        values.pop("transform_spec_sha256")
        values["transform_spec"] = _transform_from_attempt_json(item)
        if values.get("input_bbox_page") is not None:
            values["input_bbox_page"] = tuple(values["input_bbox_page"])
        attempts.append(OCRAttempt(**values))
    blocks = []
    for item in payload["blocks"]:
        values = dict(item)
        values["bbox_page"] = tuple(values["bbox_page"])
        values["polygon_page"] = tuple(tuple(point) for point in values["polygon_page"])
        blocks.append(OCRBlock(**values))
    result = OCRInvocationResult.build(
        request=request,
        blocks=tuple(blocks),
        observations=tuple(_record_from_json(item) for item in payload["observations"]),
        full_page_lines=tuple(_record_from_json(item) for item in payload["full_page_lines"]),
        attempts=tuple(attempts),
        diagnostics=OCRDiagnostics(
            payload["diagnostics"]["provider"], payload["diagnostics"].get("extras") or {}
        ),
    )
    if result.attempt_chain_sha256 != payload["attempt_chain_sha256"]:
        raise CoverageInvariantError("OCR attempt chain changed during reopen")
    return result


def _recovery_request_from_json(payload: Mapping[str, object]) -> CoverageRecoveryRequest:
    values = dict(payload)
    values["observation_ids"] = tuple(values.get("observation_ids") or ())
    values["evidence_ids"] = tuple(values.get("evidence_ids") or ())
    if values.get("anchor_polygon_page") is not None:
        values["anchor_polygon_page"] = tuple(tuple(point) for point in values["anchor_polygon_page"])
    return CoverageRecoveryRequest(**values)


def _recovery_decision_from_json(payload: Mapping[str, object]) -> CoverageRecoveryDecision:
    values = dict(payload)
    values["observation_ids"] = tuple(values.get("observation_ids") or ())
    values["evidence_ids"] = tuple(values.get("evidence_ids") or ())
    if values.get("anchor_polygon_page") is not None:
        values["anchor_polygon_page"] = tuple(tuple(point) for point in values["anchor_polygon_page"])
    return CoverageRecoveryDecision(**values)


def _ledger_from_json(wrapper: Mapping[str, object]) -> PageCoverageLedger:
    payload = wrapper["payload"]
    inventory = tuple(CoverageComponentInventoryEntry(**item) for item in payload["component_inventory"])
    entries = []
    for item in payload["entries"]:
        values = dict(item)
        values["bbox_page"] = tuple(values["bbox_page"])
        values["polygon_page"] = tuple(tuple(point) for point in values["polygon_page"])
        for name in ("ocr_attempt_ids", "observation_ids", "protection_evidence_ids"):
            values[name] = tuple(values.get(name) or ())
        entries.append(CoverageEntry(**values))
    dispositions = []
    for item in payload["observation_dispositions"]:
        values = dict(item)
        values["evidence_ids"] = tuple(values.get("evidence_ids") or ())
        dispositions.append(CoverageObservationDisposition(**values))
    ledger = build_page_coverage_ledger(
        run_id=payload["run_id"],
        origin_execution_id=payload["origin_execution_id"],
        page_id=payload["page_id"],
        page_source_sha256=payload["page_source_sha256"],
        inventory_version=int(payload["inventory_version"]),
        parent_ledger_sha256=payload.get("parent_ledger_sha256"),
        component_inventory=inventory,
        expected_observation_ids=tuple(payload["expected_observation_ids"]),
        entries=tuple(entries),
        observation_dispositions=tuple(dispositions),
        recovery_requests=tuple(_recovery_request_from_json(item) for item in payload["recovery_requests"]),
        recovery_decisions=tuple(_recovery_decision_from_json(item) for item in payload["recovery_decisions"]),
    )
    if ledger.sha256 != wrapper["sha256"]:
        raise CoverageInvariantError("coverage ledger hash changed during reopen")
    return ledger


def _bbox_overlap_fraction(left: BBox, right: BBox) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / float(max(1, (left[2] - left[0]) * (left[3] - left[1])))


CoverageOCRRunner = Callable[..., OCRInvocationResult]


def classify_empty_primary_candidate(
    component: "SourceTextComponent",
    *,
    has_coverage_observation: bool,
    has_explicit_preserve_policy: bool,
) -> bool:
    """Return whether an unsupported primary hypothesis needs human review.

    This does not assert non-text. A valid OCR observation, glyph/script support,
    or an audited preservation policy takes precedence over the uncertainty rule.
    Detector confidence remains evidence for diagnostics, but cannot establish
    translatable materiality without independent semantic corroboration.
    """

    return (
        not has_coverage_observation
        and not has_explicit_preserve_policy
        and frozenset(component.detector_sources)
        == frozenset(("primary_region_detector",))
        and not component.script_evidence
    )


def _ocr_empty_component_preserve_policy(
    page_rgb: "np.ndarray",
    component: "SourceTextComponent",
    *,
    page_observations: Sequence[Any] = (),
) -> tuple[str, str] | None:
    """Classify OCR-empty hypotheses that have no independent corroboration.

    OCR exhaustion alone is never enough to preserve a component.  This
    disposition additionally requires either an uncorroborated glyph scan or
    an evidence-bound dark-container heuristic without primary/script
    confirmation.  Specific low-confidence visual shapes retain narrower
    audit policy identifiers.
    """

    import numpy as np

    detector_sources = frozenset(component.detector_sources)
    region_sources = frozenset(
        (
            "primary_region_detector",
            "negative_region_detector",
            "strip_region_detector",
        )
    )
    page_height, page_width = page_rgb.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in component.bbox_page)
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    normalized_page_text = " ".join(
        re.sub(r"[^A-Z0-9]+", " ", str(getattr(item, "text", "")).upper()).strip()
        for item in page_observations
    )
    credit_markers = {
        marker
        for marker in (
            "RESET SCAN",
            "TOON",
            "FASTER UPDATE",
            "PATREON",
            "DISCORD",
            "RECRUIT",
            "TRANSLATOR",
            "EDITOR",
        )
        if marker in normalized_page_text
    }
    corroborated_region = (
        "glyph_scan" in detector_sources
        and bool(detector_sources & region_sources)
        and not component.script_evidence
        and bool(component.evidence_ids)
    )
    regional_credit_candidate = (
        bool(detector_sources & region_sources)
        and not component.script_evidence
        and bool(component.evidence_ids)
    )
    edge_credit_banner = (
        regional_credit_candidate
        and len(credit_markers) >= 2
        and (
            y2 <= page_height * 0.12
            or y1 >= page_height * 0.96
        )
        and box_width >= page_width * 0.45
        and box_height <= page_height * 0.04
        and box_width >= 1.4 * max(1, box_height)
    )
    if edge_credit_banner or (corroborated_region and (
        (
            len(credit_markers) >= 2
            and y2 <= page_height * 0.12
            and box_width >= page_width * 0.30
            and box_height <= page_height * 0.03
            and box_width >= 3 * max(1, box_height)
        )
        or (
            len(credit_markers) >= 3
            and page_height <= page_width * 2.5
        )
    )):
        return ("visual_non_text", "policy:scanlation_credit_art")
    if (
        detector_sources & region_sources
        and detector_sources <= region_sources | frozenset(("glyph_scan",))
    ):
        try:
            from vision_stack.sfx_detector import text_blocks_to_sfx_candidates
        except ImportError:  # pragma: no cover - package import fallback
            from ..vision_stack.sfx_detector import text_blocks_to_sfx_candidates

        sfx_candidates = text_blocks_to_sfx_candidates(
            page_rgb,
            [
                {
                    "bbox": list(component.bbox_page),
                    "confidence": float(component.confidence or 0.0),
                }
            ],
            source="owner_coverage_region_detector",
            min_confidence=0.01,
            min_area_ratio=0.0,
        )
        if sfx_candidates:
            return (
                "sfx",
                "policy:explicit_sfx_outside_translatable_container",
            )
        if (
            corroborated_region
            and box_width <= page_width * 0.18
            and box_height <= page_height * 0.025
            and box_width * box_height <= page_width * page_height * 0.004
        ):
            return (
                "sfx",
                "policy:ocr_empty_small_corroborated_sfx",
            )
    if (
        detector_sources
        and detector_sources <= region_sources
        and not component.script_evidence
    ):
        touches_horizontal_edge = (
            y1 <= max(2, int(round(page_height * 0.005)))
            or y2 >= page_height - max(2, int(round(page_height * 0.005)))
        )
        crop = page_rgb[y1:y2, x1:x2, :3]
        repeated_edge_pattern = False
        if crop.size:
            rgb_i32 = crop.astype("int32", copy=False)
            luminance = (
                77 * rgb_i32[:, :, 0]
                + 150 * rgb_i32[:, :, 1]
                + 29 * rgb_i32[:, :, 2]
                + 128
            ) // 256
            transition_counts = (
                np.abs(np.diff(luminance, axis=1)) >= 48
            ).sum(axis=1).astype("float64")
            transition_mean = float(transition_counts.mean())
            transition_cv = float(transition_counts.std()) / max(
                1.0, transition_mean
            )
            repeated_edge_pattern = (
                float(np.median(transition_counts)) >= 10.0
                and transition_cv <= 0.45
            )
        if (
            touches_horizontal_edge
            and box_width >= page_width * 0.70
            and box_height <= page_height * 0.10
            and box_width >= 6 * max(1, box_height)
            and repeated_edge_pattern
        ):
            return (
                "visual_non_text",
                "policy:explicit_visual_non_text",
            )
    if detector_sources == frozenset(("dark_balloon_band_scan", "glyph_scan")):
        if component.script_evidence or not component.evidence_ids:
            return None
        return ("visual_non_text", "policy:explicit_visual_non_text")
    if detector_sources != frozenset(("glyph_scan",)):
        return None
    if component.script_evidence or component.evidence_ids:
        return None
    confidence = float(component.confidence)
    if (
        confidence <= 0.60
        and box_width <= 24
        and box_height <= 24
        and box_width * box_height <= 512
    ):
        return (
            "visual_non_text",
            "policy:ocr_empty_tiny_isolated_false_glyph",
        )
    crop = page_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 2:
        luminance = crop
        max_channel_span = int(crop.max()) - int(crop.min())
    else:
        rgb = crop[:, :, :3]
        pixels = rgb.reshape((-1, 3))
        channel_spans = pixels.max(axis=0) - pixels.min(axis=0)
        max_channel_span = int(channel_spans.max())
        rgb_i32 = rgb.astype("int32", copy=False)
        luminance = (
            77 * rgb_i32[:, :, 0]
            + 150 * rgb_i32[:, :, 1]
            + 29 * rgb_i32[:, :, 2]
            + 128
        ) // 256
    if (
        confidence <= 0.75
        and max_channel_span <= 24
        and int(luminance.max()) - int(luminance.min()) <= 12
    ):
        return (
            "visual_non_text",
            "policy:ocr_empty_near_uniform_false_glyph",
        )
    return (
        "visual_non_text",
        "policy:ocr_empty_uncorroborated_glyph_scan",
    )


def complete_page_coverage(
    page_rgb: "np.ndarray",
    *,
    run_id: str,
    origin_execution_id: str,
    page_id: str,
    page_source_sha256: str,
    components: tuple["SourceTextComponent", ...],
    band_evidence: Sequence[object],
    ocr_runner: CoverageOCRRunner | None = None,
) -> PageCoverageResult:
    """Perform page-global and component-anchored OCR before owner resolution."""

    from .hash_contract import canonical_page_sha256
    from .model import TextObservation

    del band_evidence
    root_hash = canonical_page_sha256(page_rgb)
    if root_hash != page_source_sha256:
        raise CoverageIdentityError("coverage page pixels differ from page_source_sha256")
    if ocr_runner is None:
        from vision_stack.runtime import run_page_coverage_ocr

        ocr_runner = run_page_coverage_ocr

    requests: list[OCRRequest] = []
    invocations: list[OCRInvocationResult] = []
    records: list[tuple[OCRObservationRecord, tuple[str, ...]]] = []

    full_request = OCRRequest(
        run_id=run_id,
        origin_execution_id=origin_execution_id,
        page_id=page_id,
        page_source_sha256=page_source_sha256,
        root_input_pixel_sha256=root_hash,
        invocation_id=f"{page_id}:coverage:full-page",
        provider_family="paddleocr",
    )
    crop_first = os.getenv("TRADUZAI_OCR_CROP_FIRST", "0").strip().lower() in {"1", "true", "yes", "on"}
    primary_components = [
        component for component in components
        if "primary_region_detector" in component.detector_sources
    ] if crop_first else []
    primary_blocks = [SimpleNamespace(xyxy=component.bbox_page) for component in primary_components]
    full_kwargs = {"coverage_blocks": primary_blocks} if primary_blocks else {}
    full = ocr_runner(page_rgb, request=full_request, bbox_page=None, variants=("full_page",), **full_kwargs)
    requests.append(full_request)
    invocations.append(full)
    full_records = tuple(full.observations or full.full_page_lines)
    crop_component_by_attempt = {
        attempt.attempt_id: component
        for attempt in full.attempts
        if attempt.input_kind in {"detected_crop", "terminal_recrop"}
        for component in primary_components
        if tuple(attempt.input_bbox_page or ()) == tuple(component.bbox_page)
    }
    associated: dict[str, list[OCRObservationRecord]] = {
        component.component_id: [] for component in components
    }
    redirected_observations: dict[str, set[str]] = {
        component.component_id: set() for component in components
    }
    superseded_full_observations: dict[str, set[str]] = {
        component.component_id: set() for component in components
    }
    for record in full_records:
        selected_component = crop_component_by_attempt.get(record.attempt_id)
        matches = []
        for component in ([] if selected_component is not None else components):
            observation_fraction = _bbox_overlap_fraction(
                record.bbox_page, component.bbox_page
            )
            if observation_fraction < 0.18:
                continue
            component_fraction = _bbox_overlap_fraction(
                component.bbox_page, record.bbox_page
            )
            component_area = max(
                1,
                (component.bbox_page[2] - component.bbox_page[0])
                * (component.bbox_page[3] - component.bbox_page[1]),
            )
            matches.append(
                (
                    observation_fraction,
                    component_fraction,
                    component_area,
                    component,
                )
            )
        if len(matches) == 1:
            selected_component = matches[0][3]
        elif len(matches) > 1:
            containing = [item for item in matches if item[0] >= 0.85]
            if containing:
                containing.sort(
                    key=lambda item: (item[2], -item[1], item[3].component_id)
                )
                best = containing[0]
                runner_up_area = containing[1][2] if len(containing) > 1 else None
                if runner_up_area is None or best[2] <= runner_up_area * 0.80:
                    selected_component = best[3]
                    for _obs_fraction, _comp_fraction, area, candidate in containing[1:]:
                        if area >= best[2] * 1.25:
                            superseded_full_observations[candidate.component_id].add(
                                selected_component.component_id
                            )
        component_ids = (
            (selected_component.component_id,)
            if selected_component is not None
            else ()
        )
        for component_id in component_ids:
            associated[component_id].append(record)
        records.append((record, component_ids))

    for component in components:
        if associated[component.component_id]:
            continue
        request = OCRRequest(
            run_id=run_id,
            origin_execution_id=origin_execution_id,
            page_id=page_id,
            page_source_sha256=page_source_sha256,
            root_input_pixel_sha256=root_hash,
            invocation_id=f"{page_id}:coverage:{component.component_id}",
            provider_family="paddleocr",
        )
        invocation = ocr_runner(
            page_rgb,
            request=request,
            bbox_page=component.bbox_page,
            variants=("native", "gray", "inverted", "scale_2x"),
        )
        requests.append(request)
        invocations.append(invocation)
        for record in invocation.observations:
            matches = []
            for candidate in components:
                observation_fraction = _bbox_overlap_fraction(
                    record.bbox_page, candidate.bbox_page
                )
                component_fraction = _bbox_overlap_fraction(
                    candidate.bbox_page, record.bbox_page
                )
                if max(observation_fraction, component_fraction) >= 0.55:
                    matches.append(
                        (
                            min(observation_fraction, component_fraction),
                            max(observation_fraction, component_fraction),
                            candidate,
                        )
                    )
            selected_component = component
            if len(matches) > 1:
                matches.sort(
                    key=lambda item: (-item[0], -item[1], item[2].component_id)
                )
                best_score, _best_overlap, best_component = matches[0]
                runner_up_score = matches[1][0]
                if best_score >= 0.45 and best_score - runner_up_score >= 0.10:
                    selected_component = best_component
            elif len(matches) == 1:
                selected_component = matches[0][2]
            associated[selected_component.component_id].append(record)
            records.append((record, (selected_component.component_id,)))
            if selected_component.component_id != component.component_id:
                redirected_observations[component.component_id].add(
                    selected_component.component_id
                )

    observations = tuple(
        TextObservation(
            observation_id=record.observation_id,
            page_id=page_id,
            component_ids=component_ids,
            text=record.text,
            confidence=record.confidence,
            provider=record.source,
            bbox_page=record.bbox_page,
            polygons_page=(record.polygon_page,) if record.polygon_page else (),
            provider_variant=record.variant_id,
            provider_record_id=record.observation_id,
            source_bbox_page=record.bbox_page,
            text_pixel_bbox_page=record.bbox_page,
            run_id=record.run_id,
            origin_execution_id=record.origin_execution_id,
            invocation_id=record.invocation_id,
            attempt_id=record.attempt_id,
            provider_family=record.provider_family,
            page_source_sha256=record.page_source_sha256,
            root_input_pixel_sha256=record.root_input_pixel_sha256,
            input_pixel_sha256=record.input_pixel_sha256,
            payload_sha256=record.payload_sha256,
        )
        for record, component_ids in records
    )
    attempts_by_invocation = {
        invocation.invocation_id: tuple(item.attempt_id for item in invocation.attempts)
        for invocation in invocations
    }
    full_attempt_ids = attempts_by_invocation[full.invocation_id]
    entries: list[CoverageEntry] = []
    for component in components:
        component_observations = tuple(associated[component.component_id])
        preserve = (
            None
            if component_observations
            else (
                ("visual_non_text", "policy:explicit_visual_non_text")
                if (
                    redirected_observations[component.component_id]
                    or superseded_full_observations[component.component_id]
                )
                and frozenset(component.detector_sources)
                <= frozenset(
                    (
                        "primary_region_detector",
                        "negative_region_detector",
                        "strip_region_detector",
                    )
                )
                else _ocr_empty_component_preserve_policy(
                    page_rgb,
                    component,
                    page_observations=observations,
                )
            )
        )
        uncertain = classify_empty_primary_candidate(
            component,
            has_coverage_observation=bool(component_observations),
            has_explicit_preserve_policy=preserve is not None,
        )
        entries.append(
            CoverageEntry(
                component_id=component.component_id,
                run_id=run_id,
                origin_execution_id=origin_execution_id,
                page_id=page_id,
                page_source_sha256=page_source_sha256,
                bbox_page=component.bbox_page,
                polygon_page=component.polygon_page,
                materiality=(
                    "non_text" if preserve else "uncertain" if uncertain else "material"
                ),
                ocr_attempt_ids=tuple(dict.fromkeys((
                    *full_attempt_ids,
                    *attempts_by_invocation.get(
                        f"{page_id}:coverage:{component.component_id}", ()
                    ),
                    *(item.attempt_id for item in component_observations),
                ))),
                observation_ids=tuple(
                    item.observation_id for item in component_observations
                ),
                semantic_role=preserve[0] if preserve else None,
                state=(
                    "observed"
                    if component_observations
                    else "explicit_non_dialogue_preserve"
                    if preserve
                    else "review_required"
                    if uncertain
                    else "challenged"
                ),
                preserve_policy=preserve[1] if preserve else None,
            )
        )
    for component, entry in zip(components, entries):
        if entry.materiality != "material" or entry.observation_ids:
            continue
        nearby = sorted(
            (
                (
                    max(
                        _bbox_overlap_fraction(observation.bbox_page, component.bbox_page),
                        _bbox_overlap_fraction(component.bbox_page, observation.bbox_page),
                    ),
                    observation.observation_id,
                    tuple(observation.bbox_page),
                    observation.text[:80],
                )
                for observation in observations
            ),
            key=lambda item: (-item[0], item[1]),
        )[:5]
        logger.warning(
            "owner coverage material component unresolved: page_id=%s component_id=%s "
            "bbox=%s detector_sources=%s nearby_observations=%s",
            page_id,
            component.component_id,
            tuple(component.bbox_page),
            tuple(component.detector_sources),
            nearby,
        )
    inventory = tuple(
        build_component_inventory_entry(
            component_id=component.component_id,
            origin="discovery",
            introduced_by_decision_id=None,
            anchor_polygon_page=component.polygon_page,
            ordinal=index,
        )
        for index, component in enumerate(components)
    )
    ledger = build_page_coverage_ledger(
        run_id=run_id,
        origin_execution_id=origin_execution_id,
        page_id=page_id,
        page_source_sha256=page_source_sha256,
        inventory_version=1,
        parent_ledger_sha256=None,
        component_inventory=inventory,
        expected_observation_ids=tuple(item.observation_id for item in observations),
        entries=tuple(entries),
    )
    return PageCoverageResult._build(
        run_id=run_id,
        origin_execution_id=origin_execution_id,
        page_id=page_id,
        page_source_sha256=page_source_sha256,
        ledger_history=(ledger,),
        components=components,
        observations=observations,
        ocr_requests=tuple(requests),
        ocr_invocations=tuple(invocations),
        recovery_requests=(),
        recovery_decisions=(),
    )


def recover_unassociated_observations(
    page_rgb: "np.ndarray",
    coverage: PageCoverageResult,
    *,
    materialize_ambiguous: bool = False,
) -> PageCoverageResult:
    from .hash_contract import canonical_page_sha256
    from .model import SourceTextComponent

    if canonical_page_sha256(page_rgb) != coverage.page_source_sha256:
        raise CoverageIdentityError("association recovery received different page pixels")
    unassociated = [item for item in coverage.observations if not item.component_ids]
    if not unassociated:
        return coverage

    observations = list(coverage.observations)
    components = list(coverage.components)
    recovery_requests = list(coverage.recovery_requests)
    recovery_decisions = list(coverage.recovery_decisions)
    recovery_request_ids = {item.request_id for item in recovery_requests}
    existing_recovery_observations = {
        observation_id
        for request in recovery_requests
        for observation_id in request.observation_ids
    }
    pending_recovery_by_observation = {
        observation_id: request
        for request in coverage.pending_requests
        for observation_id in request.observation_ids
    }

    def polygon_for(observation: "TextObservation") -> Polygon:
        if observation.polygons_page and observation.polygons_page[0]:
            return tuple(observation.polygons_page[0])
        x1, y1, x2, y2 = observation.bbox_page
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    def matching_components(observation: "TextObservation") -> list["SourceTextComponent"]:
        matches = []
        for component in components:
            observation_fraction = _bbox_overlap_fraction(
                observation.bbox_page, component.bbox_page
            )
            component_fraction = _bbox_overlap_fraction(
                component.bbox_page, observation.bbox_page
            )
            if max(observation_fraction, component_fraction) >= 0.55:
                matches.append(component)
        return matches

    def request_for(observation: "TextObservation", *, attempt_kind: str):
        polygon = polygon_for(observation)
        return build_recovery_request(
            request_id=(
                "coverage_request_"
                + canonical_json_sha256(
                    {
                        "attempt_kind": attempt_kind,
                        "observation_id": observation.observation_id,
                        "page_source_sha256": coverage.page_source_sha256,
                    }
                )[:20]
            ),
            run_id=coverage.run_id,
            origin_execution_id=coverage.origin_execution_id,
            page_id=coverage.page_id,
            page_source_sha256=coverage.page_source_sha256,
            component_id=None,
            observation_ids=(observation.observation_id,),
            anchor_polygon_page=polygon,
            attempt_kind=attempt_kind,
            transform_spec_sha256=canonical_json_sha256(
                {"algorithm": attempt_kind, "schema_version": 1}
            ),
            geometry_sha256=canonical_json_sha256(
                {"polygon_page": [list(point) for point in polygon]}
            ),
            input_pixel_sha256=observation.input_pixel_sha256,
            reason="unassociated_full_page_observation",
            evidence_ids=(observation.observation_id,),
            parent_decision=None,
            next_strategy=None,
        )

    changed_association = False
    materialize_observations: list["TextObservation"] = []
    ambiguous_materialization_ids: set[str] = set()
    materialization_requests: dict[str, CoverageRecoveryRequest] = {}
    for observation in unassociated:
        matches = matching_components(observation)
        index = observations.index(observation)
        if len(matches) == 1:
            observations[index] = replace(
                observation, component_ids=(matches[0].component_id,)
            )
            changed_association = True
            continue
        if len(matches) > 1:
            def _match_score(component: "SourceTextComponent"):
                observation_fraction = _bbox_overlap_fraction(
                    observation.bbox_page, component.bbox_page
                )
                component_fraction = _bbox_overlap_fraction(
                    component.bbox_page, observation.bbox_page
                )
                return (
                    min(observation_fraction, component_fraction),
                    max(observation_fraction, component_fraction),
                    component,
                )

            scored_matches = sorted(
                (_match_score(component) for component in matches),
                key=lambda item: (-item[0], -item[1], item[2].component_id),
            )
            best_score, _best_overlap, best_component = scored_matches[0]
            runner_up_score = scored_matches[1][0]
            if best_score >= 0.45 and best_score - runner_up_score >= 0.10:
                request = request_for(
                    observation,
                    attempt_kind="anchored_association_recovery",
                )
                decision = build_recovery_decision(
                    decision_id=f"decision_{request.request_sha256[:20]}",
                    request=request,
                    materialized_component_id=None,
                    attempt_id=observation.attempt_id,
                    status="succeeded",
                    reason="associated_to_decisive_geometry_match",
                    evidence_ids=(observation.observation_id,),
                    next_strategy=None,
                )
                recovery_requests.append(request)
                recovery_decisions.append(decision)
                observations[index] = replace(
                    observation,
                    component_ids=(best_component.component_id,),
                )
                changed_association = True
                continue
            if materialize_ambiguous:
                materialize_observations.append(observation)
                ambiguous_materialization_ids.add(observation.observation_id)
                existing_request = pending_recovery_by_observation.get(
                    observation.observation_id
                )
                if existing_request is not None:
                    materialization_requests[observation.observation_id] = existing_request
                continue
            if observation.observation_id not in existing_recovery_observations:
                recovery_requests.append(
                    request_for(observation, attempt_kind="anchored_association_recovery")
                )
            continue
        materialize_observations.append(observation)

    current = coverage
    if changed_association:
        current = PageCoverageResult.build_from(
            current,
            observations=tuple(observations),
        )

    for original_observation in materialize_observations:
        observation = next(
            item
            for item in observations
            if item.observation_id == original_observation.observation_id
        )
        index = observations.index(observation)

        is_ambiguous_materialization = (
            observation.observation_id in ambiguous_materialization_ids
        )
        request = materialization_requests.get(observation.observation_id)
        if request is None:
            request = request_for(
                observation,
                attempt_kind=(
                    "ambiguous_observation_materialization"
                    if is_ambiguous_materialization
                    else "unassociated_observation_materialization"
                ),
            )
        polygon = polygon_for(observation)
        component_id = "component_" + canonical_json_sha256(
            {
                "page_source_sha256": coverage.page_source_sha256,
                "polygon_page": [list(point) for point in polygon],
            }
        )[:24]
        decision = build_recovery_decision(
            decision_id=f"decision_{request.request_sha256[:20]}",
            request=request,
            materialized_component_id=component_id,
            attempt_id=observation.attempt_id,
            status="succeeded",
            reason=(
                "materialized_from_ambiguous_full_page_observation"
                if is_ambiguous_materialization
                else "materialized_from_full_page_observation"
            ),
            evidence_ids=(observation.observation_id,),
            next_strategy=None,
        )
        component = SourceTextComponent(
            component_id=component_id,
            page_id=coverage.page_id,
            bbox_page=observation.bbox_page,
            polygon_page=polygon,
            detector_sources=("ocr_full_page_materialization",),
            confidence=observation.confidence,
            evidence_ids=(observation.observation_id,),
        )
        inventory_entry = build_component_inventory_entry(
            component_id=component_id,
            origin="recovery_materialization",
            introduced_by_decision_id=decision.decision_id,
            anchor_polygon_page=polygon,
            ordinal=len(current.ledger.component_inventory),
            decision=decision,
        )
        # A full-page OCR/glyph-confirmed region that discovery omitted is
        # conservatively treated as a protection conflict.  Downstream R2/R3
        # may clean the confirmed glyph pixels while retaining this evidence
        # to keep surrounding art protected.
        protection_conflict = True
        entry = CoverageEntry(
            component_id=component_id,
            run_id=coverage.run_id,
            origin_execution_id=coverage.origin_execution_id,
            page_id=coverage.page_id,
            page_source_sha256=coverage.page_source_sha256,
            bbox_page=observation.bbox_page,
            polygon_page=polygon,
            materiality="material",
            ocr_attempt_ids=(observation.attempt_id,),
            observation_ids=(observation.observation_id,),
            protection_conflict=protection_conflict,
            protection_evidence_ids=(observation.observation_id,),
            state="observed",
        )
        if request.request_id not in recovery_request_ids:
            recovery_requests.append(request)
            recovery_request_ids.add(request.request_id)
        recovery_decisions.append(decision)
        components.append(component)
        observations[index] = replace(
            observation, component_ids=(component_id,)
        )
        ledger = build_page_coverage_ledger(
            run_id=coverage.run_id,
            origin_execution_id=coverage.origin_execution_id,
            page_id=coverage.page_id,
            page_source_sha256=coverage.page_source_sha256,
            inventory_version=current.ledger.inventory_version + 1,
            parent_ledger_sha256=current.ledger.sha256,
            component_inventory=(*current.ledger.component_inventory, inventory_entry),
            expected_observation_ids=tuple(
                item.observation_id for item in observations
            ),
            entries=(*current.ledger.entries, entry),
            observation_dispositions=current.ledger.observation_dispositions,
            recovery_requests=tuple(recovery_requests),
            recovery_decisions=tuple(recovery_decisions),
        )
        validate_inventory_successor(current.ledger, ledger)
        current = PageCoverageResult.build_from(
            current,
            ledger_history=(*current.ledger_history, ledger),
            components=tuple(components),
            observations=tuple(observations),
            recovery_requests=tuple(recovery_requests),
            recovery_decisions=tuple(recovery_decisions),
        )

    if tuple(recovery_requests) == current.recovery_requests:
        return current
    return PageCoverageResult.build_from(
        current,
        observations=tuple(observations),
        recovery_requests=tuple(recovery_requests),
        recovery_decisions=tuple(recovery_decisions),
    )


def complete_container_coverage(
    page_rgb: "np.ndarray", coverage: PageCoverageResult
) -> PageCoverageResult:
    from .container_evidence import (
        canonical_component_container_ids,
        recover_component_visual_container,
    )
    from .hash_contract import canonical_page_sha256

    if canonical_page_sha256(page_rgb) != coverage.page_source_sha256:
        raise CoverageIdentityError("container recovery received different page pixels")
    entries: list[CoverageEntry] = []
    evidence_by_component: dict[str, dict[str, object]] = {}
    observations_by_id = {
        observation.observation_id: observation
        for observation in coverage.observations
    }
    changed = False
    for entry in coverage.entries:
        if (
            entry.container_id
            or not entry.ocr_attempt_ids
            or entry.state == "explicit_non_dialogue_preserve"
        ):
            entries.append(entry)
            continue
        component_observations = [
            observations_by_id[observation_id]
            for observation_id in entry.observation_ids
            if observation_id in observations_by_id
        ]
        if component_observations:
            semantic_bbox = (
                min(item.bbox_page[0] for item in component_observations),
                min(item.bbox_page[1] for item in component_observations),
                max(item.bbox_page[2] for item in component_observations),
                max(item.bbox_page[3] for item in component_observations),
            )
            semantic_polygon = (
                (semantic_bbox[0], semantic_bbox[1]),
                (semantic_bbox[2], semantic_bbox[1]),
                (semantic_bbox[2], semantic_bbox[3]),
                (semantic_bbox[0], semantic_bbox[3]),
            )
        else:
            semantic_bbox = entry.bbox_page
            semantic_polygon = entry.polygon_page
        evidence = recover_component_visual_container(
            page_rgb,
            component_id=entry.component_id,
            glyph_bbox_page=semantic_bbox,
            glyph_polygon_page=semantic_polygon,
        )
        evidence_by_component[entry.component_id] = evidence
        entries.append(entry)
        changed = True
    if not changed:
        return coverage
    container_ids = canonical_component_container_ids(evidence_by_component)
    entries = [
        replace(entry, container_id=container_ids[entry.component_id])
        if entry.component_id in container_ids
        else entry
        for entry in entries
    ]
    ledger = build_page_coverage_ledger(
        run_id=coverage.run_id,
        origin_execution_id=coverage.origin_execution_id,
        page_id=coverage.page_id,
        page_source_sha256=coverage.page_source_sha256,
        inventory_version=coverage.ledger.inventory_version + 1,
        parent_ledger_sha256=coverage.ledger.sha256,
        component_inventory=coverage.ledger.component_inventory,
        expected_observation_ids=coverage.ledger.expected_observation_ids,
        entries=tuple(entries),
        observation_dispositions=coverage.ledger.observation_dispositions,
        recovery_requests=coverage.recovery_requests,
        recovery_decisions=coverage.recovery_decisions,
    )
    return PageCoverageResult.build_from(
        coverage,
        ledger_history=(*coverage.ledger_history, ledger),
    )
