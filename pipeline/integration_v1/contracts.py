"""Versioned glue contracts over the existing owner pipeline objects.

This module intentionally references the current owner-specific contracts instead of
reimplementing OCR, restoration, layout, or rendering behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Literal, Mapping

from ownership.hash_contract import canonical_json_sha256


CONTRACT_VERSION = "traduzai.integration.v1"
STUDIO_IPC_VERSION = "traduzai.studio-ipc.v1"

_CONTRACTS: dict[str, dict[str, Any]] = {
    "SourceRef": {
        "owner": "integration",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "ownership.chapter_contract.SourcePageManifestEntry",
        "required_fields": ["manifest_entry", "ordinal", "orientation_degrees", "transforms"],
    },
    "AnalysisRecord": {
        "owner": "vision",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "integration_v1.contracts.AnalysisRecord",
        "adapts_to": "ownership.model.OwnerGraph",
        "statuses": ["building", "complete", "cancelled", "failed"],
        "identity_fields": [
            "source_sha256",
            "authenticated_neighbor_sha256s",
            "region",
            "coordinate_space",
            "transform_sha256",
            "source_language",
            "analysis_config_sha256",
            "provider_family",
            "provider_name",
            "provider_model",
            "provider_version",
            "capability_version",
        ],
        "reference_fields": [
            "artifact_refs",
            "transform_ref",
            "ocr_observations",
            "selected_observation_id",
            "selection_provenance",
            "logical_units",
            "physical_subblocks",
            "relations",
            "reading_order",
            "container_contour_ref",
            "writing_body_ref",
            "tail_ref",
            "dependency_hashes",
        ],
        "mask_channels": [
            "glyph",
            "outline",
            "shadow",
            "glow",
            "ignore_or_uncertain",
        ],
        "mask_channel_states": ["confirmed_present", "confirmed_empty", "unknown", "unavailable"],
        "unknown_semantics": "unknown_or_unavailable_is_not_confirmed_empty_and_has_no_zero_mask_alias",
        "publication_rule": "only_complete_revisions_are_publishable",
    },
    "TranslationUnit": {
        "owner": "integration",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "ownership.translation.TranslationBinding",
        "required_fields": ["owner_id", "source", "context", "target", "provenance", "validation", "destination_ids"],
    },
    "RestorationPlan": {
        "owner": "integration",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "inpainter.owner_mask.OwnerMaskPlan",
        "required_fields": ["owner_id", "authorized_ink", "action_mask_ref", "patch_ref", "protected_art_mask_ref", "dependency_hashes"],
    },
    "LayoutRequest": {
        "owner": "renderer",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "typesetter.backend_contract.TypesettingRenderRequest",
        "required_fields": ["owner_id", "text", "bubble_mask_path", "style", "anchors", "policies"],
    },
    "LayoutPlan": {
        "owner": "renderer",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "typesetter.style_materialization.OwnerStyleMaterializationPlan",
        "required_fields": ["owner_id", "lines", "metrics", "usable_body", "plan_sha256"],
    },
    "Recipe": {
        "owner": "renderer",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "integration_v1.contracts.RecipeReceipt",
        "required_fields": ["recipe_sha256", "output_sha256", "relative_path", "runtime_id", "runtime_sha256", "source_sha256", "dependency_hashes"],
    },
    "ReviewDecision": {
        "owner": "studio",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "integration_v1.contracts.ReviewDecision",
        "required_fields": ["project_id", "owner_id", "expected_revision", "actor_kind", "actor_id", "decision", "reason_code", "evidence_sha256s", "idempotency_key"],
    },
    "ProjectEvent": {
        "owner": "integration",
        "schema_version": 1,
        "python_type": "integration_v1.contracts.adapt_contract_payload",
        "runtime_type": "integration_v1.contracts.ProjectEvent",
        "required_fields": ["job_id", "project_id", "expected_revision", "project_revision", "sequence", "stage", "status", "reason_code", "payload"],
    },
}

for _descriptor in _CONTRACTS.values():
    _descriptor["schema_sha256"] = canonical_json_sha256(_descriptor)

_A = "a" * 64
_B = "b" * 64
_EXAMPLES: dict[str, dict[str, Any]] = {
    "SourceRef": {
        "manifest_entry": {
            "page_id": "page_001",
            "relative_source_path": "source/001.webp",
            "source_file_sha256": _A,
            "page_source_sha256": _B,
            "width": 1200,
            "height": 1800,
            "mode": "RGB",
        },
        "ordinal": 0,
        "orientation_degrees": 0,
        "transforms": [
            {"kind": "identity", "from_space": "source", "to_space": "logical_page", "inverse": "identity"}
        ],
    },
    "AnalysisRecord": {
        "status": "complete",
        "source_sha256": _B,
        "authenticated_neighbor_sha256s": [_A],
        "region": {"bbox": [100, 200, 500, 600], "coordinate_space": "logical_page"},
        "coordinate_space": "logical_page",
        "transform_sha256": _A,
        "source_language": "en",
        "analysis_config_sha256": _A,
        "provider_family": "synthetic-contract-example",
        "provider_name": "example-provider",
        "provider_model": "example-model",
        "provider_version": "1",
        "capability_version": "1",
        "artifact_refs": [{"kind": "ocr_observations", "sha256": _A}],
        "transform_ref": {"kind": "identity", "sha256": _A, "inverse_sha256": _A},
        "ocr_observations": {"artifact_ref": "analysis/ocr-observations.json", "sha256": _A, "count": 1},
        "selected_observation_id": "ocr-001",
        "selection_provenance": {"kind": "synthetic_example", "sha256": _A},
        "logical_units": {"artifact_ref": "analysis/logical-units.json", "sha256": _A, "count": 1},
        "physical_subblocks": {"artifact_ref": "analysis/physical-subblocks.json", "sha256": _A, "count": 1},
        "relations": {"artifact_ref": "analysis/relations.json", "sha256": _A, "count": 0},
        "reading_order": {"artifact_ref": "analysis/reading-order.json", "sha256": _A, "count": 1},
        "container_contour_ref": {"sha256": _A},
        "writing_body_ref": {"sha256": _A},
        "tail_ref": None,
        "dependency_hashes": {"provider": _A, "analysis_config": _B},
        "mask_channels": {
            "glyph": {"state": "confirmed_present", "artifact_ref": "analysis/masks/glyph.png", "sha256": _A},
            "outline": {"state": "confirmed_empty", "evidence_sha256": _A},
            "shadow": {"state": "unknown", "reason_code": "provider_not_run"},
            "glow": {"state": "unavailable", "reason_code": "capability_unavailable"},
            "ignore_or_uncertain": {"state": "confirmed_present", "artifact_ref": "analysis/masks/uncertain.png", "sha256": _B},
        },
    },
    "TranslationUnit": {
        "owner_id": "owner-001",
        "source": {"text": "SOURCE TEXT", "locale": "en", "observation_id": "ocr-001", "sha256": _A},
        "context": {"text": "Previous dialogue only", "publish": False, "sha256": _B},
        "target": {"text": "TEXTO TRADUZIDO", "locale": "pt-BR", "sha256": _B},
        "provenance": {"provider": "synthetic-contract-example", "attempt_ids": ["attempt-001"]},
        "validation": {"status": "accepted", "policy_version": "pt-br-v1"},
        "destination_ids": ["block-001"],
    },
    "RestorationPlan": {
        "owner_id": "owner-001",
        "authorized_ink": {"source_sha256": _B, "evidence_ids": ["mask-evidence-001"]},
        "action_mask_ref": {"relative_path": "masks/page_001/owner-001-action.png", "sha256": _A},
        "patch_ref": {"relative_path": "patches/page_001/owner-001.png", "sha256": _B},
        "protected_art_mask_ref": {"relative_path": "masks/page_001/owner-001-protected.png", "sha256": _A},
        "dependency_hashes": {"analysis": _A, "translation": _B},
    },
    "LayoutRequest": {
        "owner_id": "owner-001",
        "text": "TEXTO TRADUZIDO",
        "bubble_mask_path": "masks/page_001/owner-001.png",
        "style": {"font_family": "Anime Ace", "font_weight": "regular"},
        "anchors": {"writing_body_bbox": [100, 200, 500, 600]},
        "policies": {"font": "closed-font-map-v1", "layout": "consumer-fast-v1"},
    },
    "LayoutPlan": {
        "owner_id": "owner-001",
        "lines": ["TEXTO", "TRADUZIDO"],
        "metrics": {"font_size_px": 30, "line_advance_px": 33},
        "usable_body": {"bbox": [100, 200, 500, 600], "coordinate_space": "logical_page"},
        "plan_sha256": _A,
    },
    "Recipe": {
        "recipe_sha256": _A,
        "output_sha256": _B,
        "relative_path": "recipes/page_001/owner-001.json",
        "runtime_id": "consumer-fast-v1",
        "runtime_sha256": _A,
        "source_sha256": _B,
        "dependency_hashes": {"layout_plan": _A, "font": _B},
    },
    "ReviewDecision": {
        "project_id": "project-001",
        "owner_id": "owner-001",
        "expected_revision": 1,
        "actor_kind": "model",
        "actor_id": "model-review-v1",
        "decision": "accept_candidate",
        "reason_code": "language_contract_passed",
        "evidence_sha256s": [_A],
        "idempotency_key": "review-owner-001-r1",
    },
    "ProjectEvent": {
        "job_id": "job-001",
        "project_id": "project-001",
        "expected_revision": 1,
        "project_revision": 1,
        "sequence": 0,
        "stage": "analysis",
        "status": "running",
        "reason_code": "provider_started",
        "payload": {"owner_id": "owner-001"},
    },
}


def _canonical_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


def _require_sha256(value: Any, field: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return candidate


def _require_relative_path(value: Any, field: str) -> str:
    candidate = str(value or "")
    if not candidate or "\\" in candidate or candidate.startswith("/") or ":" in candidate:
        raise ValueError(f"{field} must be a portable relative path")
    if any(part in {"", ".", ".."} for part in candidate.split("/")):
        raise ValueError(f"{field} must not escape its root")
    return candidate


def adapt_contract_payload(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize one published contract payload.

    The adapters keep the shared envelope stable while checking the existing
    runtime object at boundaries where that object can represent the contract
    without losing information.
    """

    if name not in _CONTRACTS:
        raise ValueError(f"unknown integration contract: {name}")
    copied = _canonical_copy(payload)
    missing = [field for field in _CONTRACTS[name].get("required_fields", ()) if field not in copied]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")

    if name == "SourceRef":
        from ownership.chapter_contract import SourcePageManifestEntry

        SourcePageManifestEntry.from_dict(copied["manifest_entry"])
        if copied["ordinal"] < 0 or copied["orientation_degrees"] not in {0, 90, 180, 270}:
            raise ValueError("SourceRef ordinal/orientation is invalid")
        if not copied["transforms"] or any("inverse" not in item for item in copied["transforms"]):
            raise ValueError("SourceRef requires reversible transforms")
    elif name == "AnalysisRecord":
        required = set(_CONTRACTS[name]["identity_fields"]) | set(_CONTRACTS[name]["reference_fields"])
        if required - set(copied):
            raise ValueError("AnalysisRecord identity/reference fields are incomplete")
        if copied.get("status") not in _CONTRACTS[name]["statuses"]:
            raise ValueError("AnalysisRecord status is invalid")
        channels = copied.get("mask_channels")
        if not isinstance(channels, Mapping) or set(channels) != set(_CONTRACTS[name]["mask_channels"]):
            raise ValueError("AnalysisRecord mask channel coverage is incomplete")
        valid_states = set(_CONTRACTS[name]["mask_channel_states"])
        for channel_name, channel in channels.items():
            if not isinstance(channel, Mapping) or channel.get("state") not in valid_states:
                raise ValueError(f"AnalysisRecord mask channel state is invalid: {channel_name}")
            if channel.get("state") == "confirmed_present":
                _require_sha256(channel.get("sha256"), f"AnalysisRecord.{channel_name}.sha256")
    elif name == "TranslationUnit":
        if copied["context"].get("publish") is not False:
            raise ValueError("translation context must not be published")
        if copied["target"].get("locale") != "pt-BR" or not copied["destination_ids"]:
            raise ValueError("TranslationUnit target or destinations are invalid")
        _require_sha256(copied["source"].get("sha256"), "TranslationUnit.source.sha256")
        _require_sha256(copied["target"].get("sha256"), "TranslationUnit.target.sha256")
    elif name == "RestorationPlan":
        for field in ("action_mask_ref", "patch_ref", "protected_art_mask_ref"):
            _require_relative_path(copied[field].get("relative_path"), f"RestorationPlan.{field}")
            _require_sha256(copied[field].get("sha256"), f"RestorationPlan.{field}.sha256")
    elif name == "LayoutRequest":
        from typesetter.backend_contract import TypesettingRenderRequest

        TypesettingRenderRequest.from_mapping(copied)
    elif name == "LayoutPlan":
        _require_sha256(copied["plan_sha256"], "LayoutPlan.plan_sha256")
        if not copied["lines"] or not copied["usable_body"].get("bbox"):
            raise ValueError("LayoutPlan requires lines and usable body")
    elif name == "Recipe":
        _require_relative_path(copied["relative_path"], "Recipe.relative_path")
        for field in ("recipe_sha256", "output_sha256", "runtime_sha256", "source_sha256"):
            _require_sha256(copied[field], f"Recipe.{field}")
        RecipeReceipt(
            recipe_sha256=copied["recipe_sha256"],
            output_sha256=copied["output_sha256"],
            relative_path=copied["relative_path"],
            runtime_id=copied["runtime_id"],
        )
    elif name == "ReviewDecision":
        ReviewDecision.build(
            project_id=copied["project_id"], owner_id=copied["owner_id"],
            expected_revision=copied["expected_revision"], actor_kind=copied["actor_kind"],
            actor_id=copied["actor_id"], decision=copied["decision"],
            reason_code=copied["reason_code"], evidence_sha256s=tuple(copied["evidence_sha256s"]),
            idempotency_key=copied["idempotency_key"],
        )
    elif name == "ProjectEvent":
        ProjectEvent.build(
            job_id=copied["job_id"], project_id=copied["project_id"],
            expected_revision=copied["expected_revision"], project_revision=copied["project_revision"],
            sequence=copied["sequence"], stage=copied["stage"], status=copied["status"],
            reason_code=copied["reason_code"], payload=copied["payload"],
        )
    return copied


def contract_snapshot() -> dict[str, Any]:
    """Return the stable registry consumed by the bootstrap publication."""

    return {
        "schema": CONTRACT_VERSION,
        "contracts": _canonical_copy(_CONTRACTS),
        "examples": {name: adapt_contract_payload(name, payload) for name, payload in _EXAMPLES.items()},
        "compatibility": {
            "project_json": "v12 aliases remain adapter-only; no second project writer",
            "consumer_fast": "runtime modules resolve from pipeline/consumer_fast/runtime",
        },
    }


@dataclass(frozen=True)
class AnalysisRecord:
    payload: dict[str, Any]
    analysis_record_sha256: str

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> "AnalysisRecord":
        adapted = adapt_contract_payload("AnalysisRecord", payload)
        return cls(
            payload=adapted,
            analysis_record_sha256=canonical_json_sha256(adapted),
        )

    def to_dict(self) -> dict[str, Any]:
        return _canonical_copy(self.payload) | {
            "analysis_record_sha256": self.analysis_record_sha256,
        }


@dataclass(frozen=True)
class ProjectEvent:
    job_id: str
    project_id: str
    expected_revision: int
    project_revision: int
    sequence: int
    stage: str
    status: str
    reason_code: str
    payload: dict[str, Any]
    payload_sha256: str
    event_id: str

    @classmethod
    def build(
        cls,
        *,
        job_id: str,
        project_id: str,
        expected_revision: int,
        project_revision: int,
        sequence: int,
        stage: str,
        status: str,
        reason_code: str,
        payload: Mapping[str, Any],
    ) -> "ProjectEvent":
        if not job_id or not project_id or not stage or not status:
            raise ValueError("project event identity is incomplete")
        if min(expected_revision, project_revision, sequence) < 0:
            raise ValueError("project event revisions and sequence must be non-negative")
        if project_revision < expected_revision:
            raise ValueError("project event revision is older than expected")
        copied = _canonical_copy(payload)
        payload_sha256 = canonical_json_sha256(copied)
        body = {
            "job_id": job_id,
            "project_id": project_id,
            "expected_revision": expected_revision,
            "project_revision": project_revision,
            "sequence": sequence,
            "stage": stage,
            "status": status,
            "reason_code": reason_code,
            "payload_sha256": payload_sha256,
        }
        return cls(
            **body,
            payload=copied,
            event_id=f"project-event:{canonical_json_sha256(body)[:32]}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobCommand:
    action: Literal["start", "cancel", "pause", "resume", "retry"]
    project_path: str
    expected_revision: int
    idempotency_key: str
    job_id: str | None
    command_id: str

    @classmethod
    def build(
        cls,
        *,
        action: Literal["start", "cancel", "pause", "resume", "retry"],
        project_path: str,
        expected_revision: int,
        idempotency_key: str,
        job_id: str | None = None,
    ) -> "JobCommand":
        if action not in {"start", "cancel", "pause", "resume", "retry"}:
            raise ValueError("unsupported job action")
        if not project_path or not idempotency_key or expected_revision < 0:
            raise ValueError("job command identity is incomplete")
        if action != "start" and not job_id:
            raise ValueError("job_id is required after start")
        body = {
            "action": action,
            "project_path": project_path,
            "expected_revision": expected_revision,
            "idempotency_key": idempotency_key,
            "job_id": job_id,
        }
        return cls(**body, command_id=f"job-command:{canonical_json_sha256(body)[:32]}")


@dataclass(frozen=True)
class ReviewDecision:
    project_id: str
    owner_id: str
    expected_revision: int
    actor_kind: Literal["human", "model", "system"]
    actor_id: str
    decision: str
    reason_code: str
    evidence_sha256s: tuple[str, ...]
    idempotency_key: str
    decision_id: str

    @property
    def is_human(self) -> bool:
        return self.actor_kind == "human"

    @classmethod
    def build(
        cls,
        *,
        project_id: str,
        owner_id: str,
        expected_revision: int,
        actor_kind: Literal["human", "model", "system"],
        actor_id: str,
        decision: str,
        reason_code: str,
        evidence_sha256s: tuple[str, ...],
        idempotency_key: str,
    ) -> "ReviewDecision":
        if actor_kind not in {"human", "model", "system"}:
            raise ValueError("review actor kind is invalid")
        if not all((project_id, owner_id, actor_id, decision, reason_code, idempotency_key)):
            raise ValueError("review decision identity is incomplete")
        if expected_revision < 0 or not evidence_sha256s:
            raise ValueError("review decision requires revision and evidence")
        if any(len(value) != 64 for value in evidence_sha256s):
            raise ValueError("review decision evidence hash is invalid")
        body = {
            "project_id": project_id,
            "owner_id": owner_id,
            "expected_revision": expected_revision,
            "actor_kind": actor_kind,
            "actor_id": actor_id,
            "decision": decision,
            "reason_code": reason_code,
            "evidence_sha256s": list(evidence_sha256s),
            "idempotency_key": idempotency_key,
        }
        return cls(
            project_id=project_id,
            owner_id=owner_id,
            expected_revision=expected_revision,
            actor_kind=actor_kind,
            actor_id=actor_id,
            decision=decision,
            reason_code=reason_code,
            evidence_sha256s=tuple(evidence_sha256s),
            idempotency_key=idempotency_key,
            decision_id=f"review-decision:{canonical_json_sha256(body)[:32]}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecipeReceipt:
    recipe_sha256: str
    output_sha256: str
    relative_path: str
    runtime_id: str


@dataclass(frozen=True)
class ExportDecision:
    allowed: bool
    status: str
    gate_allowed: bool
    verified: bool
    completion_status: str
    output_review_state: str
    blocker_count: int
    critical_issue_count: int
    critical_flag_count: int
    review_issue_count: int
    review_flag_count: int

    @classmethod
    def from_project(cls, project: Mapping[str, Any]) -> "ExportDecision":
        qa = project.get("qa") if isinstance(project.get("qa"), Mapping) else {}
        gate = qa.get("export_gate") if isinstance(qa.get("export_gate"), Mapping) else {}
        status = str(gate.get("status") or "BLOCK")
        gate_allowed = gate.get("allowed") is True
        verified = bool(project.get("verified"))
        completion_status = str(project.get("completion_status") or "")
        output_review_state = str(project.get("output_review_state") or "")
        blocker_count = int(project.get("blocker_count") or gate.get("blocker_count") or 0)
        critical_issue_count = int(gate.get("critical_issue_count") or 0)
        critical_flag_count = int(gate.get("critical_flag_count") or 0)
        review_issue_count = int(gate.get("review_issue_count") or 0)
        review_flag_count = int(gate.get("review_flag_count") or 0)
        issue_count = int(gate.get("issue_count") or len(gate.get("issues") or ()))
        allowed = bool(
            verified
            and completion_status == "approved"
            and output_review_state == "approved"
            and status == "PASS"
            and gate_allowed
            and blocker_count == 0
            and critical_issue_count == 0
            and critical_flag_count == 0
            and review_issue_count == 0
            and review_flag_count == 0
            and issue_count == 0
        )
        return cls(
            allowed=allowed,
            status=status,
            gate_allowed=gate_allowed,
            verified=verified,
            completion_status=completion_status,
            output_review_state=output_review_state,
            blocker_count=blocker_count,
            critical_issue_count=critical_issue_count,
            critical_flag_count=critical_flag_count,
            review_issue_count=review_issue_count,
            review_flag_count=review_flag_count,
        )


def studio_ipc_contract() -> dict[str, Any]:
    mutation_fields = ["expected_revision", "idempotency_key"]
    return {
        "schema": STUDIO_IPC_VERSION,
        "contract_owner": "integration",
        "transport_adapter_owner": "studio",
        "implementation_state": "awaiting_specialist_handoff",
        "commands": {
            "start_consumer_fast": {
                "arguments": ["project_path", "chapter_id", *mutation_fields],
                "returns": ["job_id", "project_revision"],
            },
            "cancel_consumer_fast": {
                "arguments": ["job_id", *mutation_fields],
                "returns": ["job_id", "project_revision", "status"],
            },
            "pause_consumer_fast": {
                "arguments": ["job_id", *mutation_fields],
                "returns": ["job_id", "project_revision", "status"],
            },
            "resume_consumer_fast": {
                "arguments": ["job_id", *mutation_fields],
                "returns": ["job_id", "project_revision", "status"],
            },
            "retry_consumer_fast": {
                "arguments": ["job_id", "terminal_unit_ids", *mutation_fields],
                "returns": ["job_id", "project_revision", "status"],
            },
            "persist_project_event": {
                "arguments": ["event", *mutation_fields],
                "returns": ["event_id", "project_revision"],
            },
            "retypeset_owner": {
                "arguments": ["project_path", "owner_id", "layout_request", *mutation_fields],
                "returns": ["project_revision", "recipe_receipt", "raster_artifact_ref"],
            },
            "decide_export": {
                "arguments": ["project_path", "expected_revision"],
                "returns": ["export_decision"],
            },
            "export_final": {
                "arguments": ["project_path", "destination", *mutation_fields],
                "returns": ["project_revision", "export_manifest", "publication_receipt"],
                "precondition": "server_rechecks_export_decision_allowed",
            },
            "export_diagnostic": {
                "arguments": ["project_path", "destination", "expected_revision"],
                "returns": ["diagnostic_manifest"],
                "precondition": "never_promotes_final_export_state",
            },
        },
        "event": {
            "type": "ProjectEvent",
            "monotonic_fields": ["job_id", "project_revision", "sequence"],
            "stale_event_policy": "discard_and_refresh",
        },
        "conflict": {
            "code": "PROJECT_REVISION_CONFLICT",
            "recovery": "reload_rebase_retry",
        },
    }
