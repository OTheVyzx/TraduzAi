from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path

import pytest


def test_contract_registry_reuses_existing_runtime_types() -> None:
    from integration_v1.contracts import (
        CONTRACT_VERSION,
        adapt_contract_payload,
        contract_snapshot,
    )

    snapshot = contract_snapshot()

    assert CONTRACT_VERSION == "traduzai.integration.v1"
    assert snapshot["schema"] == CONTRACT_VERSION
    assert set(snapshot["contracts"]) == {
        "SourceRef",
        "AnalysisRecord",
        "TranslationUnit",
        "RestorationPlan",
        "LayoutRequest",
        "LayoutPlan",
        "Recipe",
        "ReviewDecision",
        "ProjectEvent",
    }
    for item in snapshot["contracts"].values():
        assert item["owner"] in {"integration", "vision", "renderer", "studio"}
        assert item["schema_version"] >= 1
        assert item["python_type"]
        assert item["runtime_type"]

    analysis = snapshot["contracts"]["AnalysisRecord"]
    assert analysis["statuses"] == ["building", "complete", "cancelled", "failed"]
    assert set(analysis["mask_channels"]) == {
        "glyph",
        "outline",
        "shadow",
        "glow",
        "ignore_or_uncertain",
    }
    assert "source_sha256" in analysis["identity_fields"]
    assert "transform_sha256" in analysis["identity_fields"]
    assert analysis["runtime_type"] == "integration_v1.contracts.AnalysisRecord"
    assert "dependency_hashes" in analysis["reference_fields"]
    assert len(analysis["schema_sha256"]) == 64

    assert set(snapshot["examples"]) == set(snapshot["contracts"])
    assert "N:\\" not in repr(snapshot["examples"])
    for name, example in snapshot["examples"].items():
        assert adapt_contract_payload(name, example) == example
    from integration_v1.contracts import AnalysisRecord

    record = AnalysisRecord.build(snapshot["examples"]["AnalysisRecord"])
    assert len(record.analysis_record_sha256) == 64
    assert record.to_dict()["status"] == "complete"
    assert set(
        channel["state"]
        for channel in record.payload["mask_channels"].values()
    ) <= {"confirmed_present", "confirmed_empty", "unknown", "unavailable"}
    assert snapshot["examples"]["SourceRef"]["manifest_entry"]["page_id"] == "page_001"
    assert snapshot["studio_ipc"]["argument_envelope"] == "direct_fields"
    assert snapshot["studio_ipc"]["event"]["channel"] == "consumer-fast-project-event"
    assert "start_consumer_fast" in snapshot["studio_ipc"]["commands"]


def test_project_event_is_deterministic_and_revision_bound() -> None:
    from integration_v1.contracts import ProjectEvent

    values = dict(
        job_id="job-001",
        project_id="project-001",
        expected_revision=7,
        project_revision=8,
        sequence=3,
        stage="translation",
        status="running",
        reason_code="provider_started",
        payload={"owner_id": "owner-001"},
    )
    first = ProjectEvent.build(**values)
    second = ProjectEvent.build(**values)

    assert first == second
    assert first.event_id.startswith("project-event:")
    assert first.to_dict()["payload_sha256"] == first.payload_sha256

    with pytest.raises(ValueError, match="older than expected"):
        ProjectEvent.build(**(values | {"project_revision": 6}))


def test_job_commands_are_job_scoped_and_idempotent() -> None:
    from integration_v1.contracts import JobCommand

    start = JobCommand.build(
        action="start",
        project_path="projects/ch57/project.json",
        expected_revision=4,
        idempotency_key="start-project-001-r4",
    )
    assert start.job_id is None
    assert start.command_id == JobCommand.build(
        action="start",
        project_path="projects/ch57/project.json",
        expected_revision=4,
        idempotency_key="start-project-001-r4",
    ).command_id

    with pytest.raises(ValueError, match="job_id"):
        JobCommand.build(
            action="cancel",
            project_path="projects/ch57/project.json",
            expected_revision=4,
            idempotency_key="cancel-project-001-r4",
        )


def test_review_decision_never_promotes_model_to_human() -> None:
    from integration_v1.contracts import ReviewDecision

    decision = ReviewDecision.build(
        project_id="project-001",
        owner_id="owner-001",
        expected_revision=8,
        actor_kind="model",
        actor_id="model-review-v1",
        decision="accept_candidate",
        reason_code="language_contract_passed",
        evidence_sha256s=("a" * 64,),
        idempotency_key="review-owner-001-r8",
    )

    assert decision.actor_kind == "model"
    assert decision.is_human is False
    assert decision.to_dict()["actor_kind"] == "model"


def test_export_decision_is_fail_closed() -> None:
    from integration_v1.contracts import ExportDecision

    blocked = ExportDecision.from_project(
        {
            "verified": False,
            "qa": {"export_gate": {"status": "BLOCK", "critical_issue_count": 1}},
        }
    )
    assert blocked.allowed is False
    assert blocked.status == "BLOCK"

    allowed = ExportDecision.from_project(
        {
            "verified": True,
            "completion_status": "approved",
            "output_review_state": "approved",
            "qa": {
                "export_gate": {
                    "status": "PASS",
                    "allowed": True,
                    "critical_issue_count": 0,
                    "critical_flag_count": 0,
                }
            },
        }
    )
    assert allowed.allowed is True

    explicit_block = ExportDecision.from_project(
        {
            "verified": True,
            "completion_status": "approved",
            "output_review_state": "approved",
            "blocker_count": 9,
            "qa": {"export_gate": {"status": "PASS", "allowed": False}},
        }
    )
    assert explicit_block.allowed is False


def test_studio_ipc_contract_lists_real_job_and_export_commands() -> None:
    from integration_v1.contracts import studio_ipc_contract

    contract = studio_ipc_contract()
    assert contract["schema"] == "traduzai.studio-ipc.v1"
    assert set(contract["commands"]) == {
        "start_consumer_fast",
        "cancel_consumer_fast",
        "pause_consumer_fast",
        "resume_consumer_fast",
        "retry_consumer_fast",
        "persist_project_event",
        "retypeset_owner",
        "decide_export",
        "export_final",
        "export_diagnostic",
    }
    assert contract["event"]["monotonic_fields"] == [
        "job_id",
        "project_revision",
        "sequence",
    ]


def test_operational_launcher_uses_only_local_runtime_roots_and_explicit_font() -> None:
    from consumer_fast import run_operational_closure

    worktree = Path(__file__).resolve().parents[2]
    roots = run_operational_closure.runtime_roots()

    assert roots
    assert all(path.resolve().is_relative_to(worktree) for path in roots)
    assert "font_path" in inspect.signature(run_operational_closure.run).parameters
    source = Path(run_operational_closure.__file__).read_text(encoding="utf-8")
    assert "N:\\TraduzAI" not in source


def test_recipe_runtime_has_no_cross_worktree_constants() -> None:
    import consumer_recipe_runtime

    worktree = Path(__file__).resolve().parents[2]
    assert consumer_recipe_runtime.PIPELINE.resolve().is_relative_to(worktree)
    assert consumer_recipe_runtime.CURRENT_OVERLAY.resolve().is_relative_to(worktree)
    source = Path(consumer_recipe_runtime.__file__).read_text(encoding="utf-8")
    assert "N:\\TraduzAI" not in source


def test_materialized_consumer_base_is_hash_bound() -> None:
    worktree = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (worktree / "pipeline/consumer_fast/BASE_SOURCES.json").read_text(encoding="utf-8")
    )
    overrides = manifest["materialized_overrides"]
    for relative_path, source_sha256 in manifest["files"].items():
        expected = overrides.get(relative_path, source_sha256)
        assert hashlib.sha256((worktree / relative_path).read_bytes()).hexdigest() == expected


def test_contract_snapshot_export_is_canonical_and_hash_bound(tmp_path: Path) -> None:
    from integration_v1.contracts import contract_snapshot
    from integration_v1.export_snapshot import write_contract_snapshot

    destination = tmp_path / "contracts" / "traduzai.integration.v1.json"
    receipt = write_contract_snapshot(destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == contract_snapshot()
    assert receipt["path"] == str(destination)
    assert len(receipt["sha256"]) == 64
    assert destination.read_bytes().endswith(b"\n")


def test_control_rerender_rasterizes_same_recipe_exactly(tmp_path: Path) -> None:
    from integration_v1.control_rerender import run_control_rerender

    receipt = run_control_rerender(tmp_path)

    assert receipt["same_recipe_same_pixels"] is True
    assert receipt["first_output_sha256"] == receipt["second_output_sha256"]
    assert receipt["first_output_sha256"] == receipt["published_output_sha256"]
    assert len(receipt["recipe_sha256"]) == 64
    assert (tmp_path / "control-rerender.png").is_file()
