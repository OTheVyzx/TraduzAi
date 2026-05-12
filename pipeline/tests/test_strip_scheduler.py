import pipeline.strip.scheduler as scheduler

from pipeline.strip.scheduler import (
    ScheduledTask,
    build_strip_scheduler_plan,
    evaluate_scheduler_gate,
    validate_schedule,
)


def test_build_strip_scheduler_plan_uses_single_gpu_lane():
    plan = build_strip_scheduler_plan(band_count=3, page_count=2)

    assert plan.max_gpu_parallel == 1
    assert plan.max_cpu_parallel >= 1
    assert plan.task_count == 13
    assert plan.gpu_task_count == 6
    assert plan.cpu_task_count == 7
    assert plan.stage_counts == {
        "concat": 1,
        "detect": 1,
        "ocr": 3,
        "translate_batch": 1,
        "inpaint": 3,
        "typeset": 3,
        "reassemble": 1,
    }
    assert plan.execution_groups[0] == ["concat"]
    assert all(len(group) <= 1 for group in plan.gpu_groups)


def test_strip_scheduler_keeps_translate_after_all_ocr_and_inpaint_after_translate():
    plan = build_strip_scheduler_plan(band_count=2, page_count=1)
    by_id = {task.task_id: task for task in plan.tasks}

    assert by_id["translate_batch"].depends_on == ("ocr:0", "ocr:1")
    assert by_id["inpaint:0"].depends_on == ("translate_batch",)
    assert by_id["inpaint:1"].depends_on == ("translate_batch",)
    assert by_id["typeset:0"].depends_on == ("inpaint:0",)
    assert by_id["reassemble"].depends_on == ("typeset:0", "typeset:1")


def test_strip_scheduler_orders_band_tasks_by_numeric_band_index():
    plan = build_strip_scheduler_plan(band_count=12, page_count=3)
    scheduled_ids = [task_id for group in plan.execution_groups for task_id in group]

    assert [task_id for task_id in scheduled_ids if task_id.startswith("ocr:")] == [
        f"ocr:{index}" for index in range(12)
    ]
    assert [task_id for task_id in scheduled_ids if task_id.startswith("inpaint:")] == [
        f"inpaint:{index}" for index in range(12)
    ]
    assert [task_id for task_id in scheduled_ids if task_id.startswith("typeset:")] == [
        f"typeset:{index}" for index in range(12)
    ]


def test_validate_schedule_fails_when_gpu_group_has_more_than_one_task():
    tasks = [
        ScheduledTask("detect", "detect", "cpu"),
        ScheduledTask("ocr:0", "ocr", "gpu", depends_on=("detect",)),
        ScheduledTask("ocr:1", "ocr", "gpu", depends_on=("detect",)),
    ]
    result = validate_schedule(tasks, execution_groups=[["detect"], ["ocr:0", "ocr:1"]], max_gpu_parallel=1)

    assert result.status == "FAIL"
    assert "GPU group exceeds single-worker limit" in result.reasons[0]


def test_scheduler_gate_writes_summary(tmp_path):
    result = evaluate_scheduler_gate(tmp_path / "gate", band_count=4, page_count=2)

    assert result["gate"]["status"] == "PASS"
    assert result["gate"]["max_gpu_parallel"] == 1
    assert result["gate"]["task_count"] == 16
    assert (tmp_path / "gate" / "summary.json").exists()


def test_process_band_stage_contract_gate_documents_monolithic_blockers(tmp_path):
    result = scheduler.evaluate_process_band_stage_contract_gate(tmp_path / "gate")

    gate = result["gate"]
    assert gate["status"] == "PASS"
    assert gate["stage_count"] == 6
    assert gate["stage_ids"] == [
        "ocr",
        "review_layout",
        "translate",
        "inpaint",
        "typeset",
        "copy_back",
    ]
    assert gate["resources"] == {
        "cpu": 3,
        "gpu": 2,
        "network": 1,
    }
    assert gate["blockers"] == []
    assert "ordered_band_context" not in gate["blockers"]
    assert "band_history" not in gate["blockers"]
    assert "running_glossary" not in gate["blockers"]
    assert "shared_gpu_models" not in gate["blockers"]
    assert "final_band_commit" not in gate["blockers"]
    assert "mutates_band_state" not in gate["blockers"]
    assert gate["next_contract_step"] == "run experimental overlap executor behind shadow gate"
    assert (tmp_path / "gate" / "summary.json").exists()


def test_gpu_ownership_gate_documents_single_gpu_lane(tmp_path):
    result = scheduler.evaluate_process_band_gpu_ownership_gate(tmp_path / "gpu_gate")

    gate = result["gate"]
    assert gate["name"] == "process_band_gpu_ownership"
    assert gate["status"] == "PASS_WITH_LIMITATION"
    assert gate["lane_id"] == "strip_single_gpu_lane"
    assert gate["max_concurrent"] == 1
    assert gate["gpu_stage_ids"] == ["ocr", "inpaint"]
    assert gate["allows_stage_overlap"] is False
    assert gate["blockers"] == []
    assert gate["limitations"] == [
        "runtime execution remains sequential",
        "ordered_band_context still serializes review/translate decisions",
    ]
    assert (tmp_path / "gpu_gate" / "summary.json").exists()


def test_ordered_context_release_gate_documents_translate_boundary(tmp_path):
    result = scheduler.evaluate_process_band_ordered_context_release_gate(tmp_path / "context_gate")

    gate = result["gate"]
    assert gate["name"] == "process_band_ordered_context_release"
    assert gate["status"] == "PASS_WITH_LIMITATION"
    assert gate["lane_id"] == "strip_ordered_context_lane"
    assert gate["ordered_stage_ids"] == ["review_layout", "translate"]
    assert gate["release_after_stage"] == "translate"
    assert gate["serializes_visual_stages"] is False
    assert gate["blockers"] == []
    assert gate["limitations"] == ["runtime execution remains sequential"]
    assert (tmp_path / "context_gate" / "summary.json").exists()
