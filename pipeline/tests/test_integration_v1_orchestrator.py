from __future__ import annotations

import json

import pytest

from integration_v1.orchestrator import (
    STAGE_ORDER, ConsumerFastJobStore, RevisionConflict, build_execution_plan,
    invalidated_stages,
)


def test_one_plan_combines_analysis_continuity_narration_and_recovery():
    plan = build_execution_plan({
        "analysis": True, "continuity": True, "narration_multiline": True,
        "operational_recovery": True,
    }, policy_versions={"vision": "v1", "language": "v1", "renderer": "v1"})
    assert plan["stages"] == list(STAGE_ORDER)
    assert plan["capabilities"] == [
        "analysis", "continuity", "narration_multiline", "operational_recovery"]
    assert len(plan["fingerprint"]) == 64
    assert plan == build_execution_plan({
        "operational_recovery": True, "analysis": True,
        "narration_multiline": True, "continuity": True,
    }, policy_versions={"renderer": "v1", "language": "v1", "vision": "v1"})


@pytest.mark.parametrize(("change", "must_include", "must_exclude"), [
    ("typography", {"layout", "rasterize", "review"}, {"ocr", "translate", "restore"}),
    ("line_breaks", {"layout", "rasterize"}, {"ocr", "translate", "restore"}),
    ("target", {"layout", "rasterize", "review"}, {"ocr", "translate"}),
    ("source_selection", {"ocr", "translate", "restore", "rasterize"}, set()),
    ("mask", {"restore", "rasterize", "review"}, {"ocr", "translate"}),
])
def test_invalidation_is_dependency_scoped(change, must_include, must_exclude):
    invalid = set(invalidated_stages(change))
    assert must_include <= invalid
    assert not (must_exclude & invalid)


def test_job_store_is_revision_bound_idempotent_cancel_resume_and_retry(tmp_path):
    store = ConsumerFastJobStore(tmp_path / "jobs")
    started = store.start(project_path="projects/ch57/project.json", expected_revision=4,
                          idempotency_key="start-r4", unit_ids=("042", "039"))
    assert started["status"] == "running" and started["project_revision"] == 5
    assert store.start(project_path="projects/ch57/project.json", expected_revision=4,
                       idempotency_key="start-r4", unit_ids=("042", "039")) == started
    paused = store.command(started["job_id"], "pause", expected_revision=5,
                           idempotency_key="pause-r5")
    assert paused["status"] == "paused"
    resumed = store.command(started["job_id"], "resume", expected_revision=6,
                            idempotency_key="resume-r6")
    assert resumed["status"] == "running"
    completed = store.finish_unit(started["job_id"], "042", "review_required",
                                  reason_code="unchanged_source_dialogue", expected_revision=7)
    assert completed["units"]["042"]["terminal"] is True
    retried = store.retry(started["job_id"], ("042",), expected_revision=8,
                          idempotency_key="retry-r8")
    assert retried["units"]["042"]["terminal"] is False
    with pytest.raises(RevisionConflict):
        store.command(started["job_id"], "cancel", expected_revision=5,
                      idempotency_key="stale-cancel")
    assert json.loads((tmp_path / "jobs" / f"{started['job_id']}.json").read_text())["status"] == "running"
