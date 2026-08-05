"""Immutable page-coverage ledger and hash-bound recovery contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from .hash_contract import canonical_json_bytes, canonical_json_sha256, sha256_bytes


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
]
CANONICAL_COVERAGE_STATES = frozenset(CoverageState.__args__)
TERMINAL_COVERAGE_STATES = frozenset(
    {"final_verified", "explicit_non_dialogue_preserve"}
)
RECOVERY_STATUSES = frozenset({"scheduled", "succeeded", "failed", "exhausted"})


class CoverageInvariantError(ValueError):
    """Raised when coverage cannot prove a complete page."""


class CoverageRecoveryExhausted(CoverageInvariantError):
    """Raised when a recovery chain reaches an exhausted decision."""


def _jsonable(value: object) -> object:
    """Convert immutable contract values to canonical-JSON containers."""

    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
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
    materiality: Literal["material", "non_text"]
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
    if entry.materiality not in {"material", "non_text"}:
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
