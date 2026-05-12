"""Pure scheduling primitives for the strip pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


ResourceKind = Literal["cpu", "gpu"]
ProcessStageResource = Literal["cpu", "gpu", "network"]

_STAGE_PRIORITY = {
    "concat": 0,
    "detect": 1,
    "ocr": 2,
    "translate_batch": 3,
    "inpaint": 4,
    "typeset": 5,
    "reassemble": 6,
}


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    stage: str
    resource: ResourceKind
    depends_on: tuple[str, ...] = ()
    page: int | None = None
    band: int | None = None


@dataclass(frozen=True)
class ScheduleValidation:
    status: str
    reasons: list[str]


@dataclass(frozen=True)
class StripSchedulerPlan:
    tasks: list[ScheduledTask]
    execution_groups: list[list[str]]
    max_cpu_parallel: int
    max_gpu_parallel: int
    validation: ScheduleValidation
    notes: list[str] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def cpu_task_count(self) -> int:
        return sum(1 for task in self.tasks if task.resource == "cpu")

    @property
    def gpu_task_count(self) -> int:
        return sum(1 for task in self.tasks if task.resource == "gpu")

    @property
    def stage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.stage] = counts.get(task.stage, 0) + 1
        return counts

    @property
    def gpu_groups(self) -> list[list[str]]:
        task_map = {task.task_id: task for task in self.tasks}
        return [
            [task_id for task_id in group if task_map[task_id].resource == "gpu"]
            for group in self.execution_groups
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "cpu_task_count": self.cpu_task_count,
            "gpu_task_count": self.gpu_task_count,
            "stage_counts": self.stage_counts,
            "max_cpu_parallel": self.max_cpu_parallel,
            "max_gpu_parallel": self.max_gpu_parallel,
            "validation": asdict(self.validation),
            "execution_groups": self.execution_groups,
            "tasks": [asdict(task) for task in self.tasks],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ProcessBandStage:
    stage_id: str
    resource: ProcessStageResource
    depends_on: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessBandStageContract:
    stages: list[ProcessBandStage]
    status: str
    blockers: list[str]
    next_contract_step: str
    notes: list[str] = field(default_factory=list)

    @property
    def stage_ids(self) -> list[str]:
        return [stage.stage_id for stage in self.stages]

    @property
    def resources(self) -> dict[str, int]:
        counts: dict[str, int] = {"cpu": 0, "gpu": 0, "network": 0}
        for stage in self.stages:
            counts[stage.resource] = counts.get(stage.resource, 0) + 1
        return {key: value for key, value in counts.items() if value}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage_count": len(self.stages),
            "stage_ids": self.stage_ids,
            "resources": self.resources,
            "blockers": list(self.blockers),
            "next_contract_step": self.next_contract_step,
            "notes": list(self.notes),
            "stages": [asdict(stage) for stage in self.stages],
        }


@dataclass(frozen=True)
class ProcessBandGpuOwnershipContract:
    lane_id: str
    max_concurrent: int
    gpu_stage_ids: tuple[str, ...]
    allows_stage_overlap: bool
    blockers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "max_concurrent": self.max_concurrent,
            "gpu_stage_ids": list(self.gpu_stage_ids),
            "allows_stage_overlap": self.allows_stage_overlap,
            "blockers": list(self.blockers),
            "limitations": list(self.limitations),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ProcessBandOrderedContextReleaseContract:
    lane_id: str
    ordered_stage_ids: tuple[str, ...]
    release_after_stage: str
    serializes_visual_stages: bool
    blockers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "ordered_stage_ids": list(self.ordered_stage_ids),
            "release_after_stage": self.release_after_stage,
            "serializes_visual_stages": self.serializes_visual_stages,
            "blockers": list(self.blockers),
            "limitations": list(self.limitations),
            "notes": list(self.notes),
        }


def build_process_band_gpu_ownership_contract() -> ProcessBandGpuOwnershipContract:
    return ProcessBandGpuOwnershipContract(
        lane_id="strip_single_gpu_lane",
        max_concurrent=1,
        gpu_stage_ids=("ocr", "inpaint"),
        allows_stage_overlap=False,
        limitations=(
            "runtime execution remains sequential",
            "ordered_band_context still serializes review/translate decisions",
        ),
        notes=(
            "OCR and inpaint may share model/runtime state only through this serialized lane.",
            "Do not schedule two GPU stages together on 8GB-class cards.",
        ),
    )


def build_process_band_ordered_context_release_contract() -> ProcessBandOrderedContextReleaseContract:
    return ProcessBandOrderedContextReleaseContract(
        lane_id="strip_ordered_context_lane",
        ordered_stage_ids=("review_layout", "translate"),
        release_after_stage="translate",
        serializes_visual_stages=False,
        limitations=("runtime execution remains sequential",),
        notes=(
            "run_chapter can merge history/glossary through ordered_context_after_translate_callback.",
            "Visual stages must not mutate the ordered context after translate.",
        ),
    )


def build_process_band_stage_contract() -> ProcessBandStageContract:
    gpu_ownership = build_process_band_gpu_ownership_contract()
    ordered_context = build_process_band_ordered_context_release_contract()
    stages = [
        ProcessBandStage(
            "ocr",
            "gpu",
            notes=(
                f"Can be replaced by precomputed_ocr_page; runtime OCR owns {gpu_ownership.lane_id}.",
            ),
        ),
        ProcessBandStage(
            "review_layout",
            "cpu",
            depends_on=("ocr",),
            notes=(f"contextual_review_page owns {ordered_context.lane_id} before translate.",),
        ),
        ProcessBandStage(
            "translate",
            "network",
            depends_on=("review_layout",),
            notes=(
                "glossary additions are released through ordered_context_after_translate_callback.",
            ),
        ),
        ProcessBandStage(
            "inpaint",
            "gpu",
            depends_on=("translate",),
            notes=(f"Inpainter shares {gpu_ownership.lane_id} with OCR; overlap is not allowed.",),
        ),
        ProcessBandStage(
            "typeset",
            "cpu",
            depends_on=("inpaint",),
            notes=("Typeset now returns a stage output snapshot before the final Band commit.",),
        ),
        ProcessBandStage(
            "copy_back",
            "cpu",
            depends_on=("typeset",),
            notes=("Copy-back returns a snapshot; the final Band commit now has an explicit serialized helper.",),
        ),
    ]
    blockers: list[str] = []
    for stage in stages:
        for blocker in stage.blockers:
            if blocker not in blockers:
                blockers.append(blocker)
    return ProcessBandStageContract(
        stages=stages,
        status="BLOCK" if blockers else "PASS",
        blockers=blockers,
        next_contract_step="run experimental overlap executor behind shadow gate",
        notes=[
            "This is a readiness gate only; it does not enable parallel execution.",
            "GPU ownership is explicit through strip_single_gpu_lane with max_concurrent=1.",
            "Ordered context is explicit through strip_ordered_context_lane and releases after translate.",
            "Final Band mutation is centralized by _commit_band_outputs and can remain serialized.",
            "Band history and glossary are captured through OrderedBandContextSnapshot.",
        ],
    )


def evaluate_process_band_gpu_ownership_gate(out_dir: str | Path | None = None) -> dict[str, Any]:
    contract = build_process_band_gpu_ownership_contract()
    status = "BLOCK" if contract.blockers else "PASS_WITH_LIMITATION"
    result = {
        "gate": {
            "name": "process_band_gpu_ownership",
            "status": status,
            "lane_id": contract.lane_id,
            "max_concurrent": contract.max_concurrent,
            "gpu_stage_ids": list(contract.gpu_stage_ids),
            "allows_stage_overlap": contract.allows_stage_overlap,
            "blockers": list(contract.blockers),
            "limitations": list(contract.limitations),
            "notes": list(contract.notes),
        },
        "contract": contract.to_dict(),
    }
    return _write_result(result, out_dir)


def evaluate_process_band_ordered_context_release_gate(out_dir: str | Path | None = None) -> dict[str, Any]:
    contract = build_process_band_ordered_context_release_contract()
    status = "BLOCK" if contract.blockers else "PASS_WITH_LIMITATION"
    result = {
        "gate": {
            "name": "process_band_ordered_context_release",
            "status": status,
            "lane_id": contract.lane_id,
            "ordered_stage_ids": list(contract.ordered_stage_ids),
            "release_after_stage": contract.release_after_stage,
            "serializes_visual_stages": contract.serializes_visual_stages,
            "blockers": list(contract.blockers),
            "limitations": list(contract.limitations),
            "notes": list(contract.notes),
        },
        "contract": contract.to_dict(),
    }
    return _write_result(result, out_dir)


def evaluate_process_band_stage_contract_gate(out_dir: str | Path | None = None) -> dict[str, Any]:
    contract = build_process_band_stage_contract()
    result = {
        "gate": {
            "name": "process_band_stage_contract",
            "status": contract.status,
            "stage_count": len(contract.stages),
            "stage_ids": contract.stage_ids,
            "resources": contract.resources,
            "blockers": list(contract.blockers),
            "next_contract_step": contract.next_contract_step,
            "notes": list(contract.notes),
        },
        "contract": contract.to_dict(),
    }
    return _write_result(result, out_dir)


def build_strip_scheduler_plan(
    *,
    band_count: int,
    page_count: int,
    max_cpu_parallel: int = 2,
    max_gpu_parallel: int = 1,
) -> StripSchedulerPlan:
    if band_count < 0:
        raise ValueError("band_count must be >= 0")
    if page_count < 1:
        raise ValueError("page_count must be >= 1")
    if max_gpu_parallel != 1:
        raise ValueError("strip scheduler currently supports exactly one GPU worker")
    if max_cpu_parallel < 1:
        raise ValueError("max_cpu_parallel must be >= 1")

    tasks: list[ScheduledTask] = [
        ScheduledTask("concat", "concat", "cpu"),
        ScheduledTask("detect", "detect", "cpu", depends_on=("concat",)),
    ]

    ocr_ids = []
    for band_index in range(band_count):
        task_id = f"ocr:{band_index}"
        ocr_ids.append(task_id)
        tasks.append(
            ScheduledTask(
                task_id,
                "ocr",
                "gpu",
                depends_on=("detect",),
                band=band_index,
                page=_page_for_band(band_index, band_count, page_count),
            )
        )

    translate_deps = tuple(ocr_ids) if ocr_ids else ("detect",)
    tasks.append(
        ScheduledTask(
            "translate_batch",
            "translate_batch",
            "cpu",
            depends_on=translate_deps,
        )
    )

    typeset_ids = []
    for band_index in range(band_count):
        inpaint_id = f"inpaint:{band_index}"
        typeset_id = f"typeset:{band_index}"
        tasks.append(
            ScheduledTask(
                inpaint_id,
                "inpaint",
                "gpu",
                depends_on=("translate_batch",),
                band=band_index,
                page=_page_for_band(band_index, band_count, page_count),
            )
        )
        tasks.append(
            ScheduledTask(
                typeset_id,
                "typeset",
                "cpu",
                depends_on=(inpaint_id,),
                band=band_index,
                page=_page_for_band(band_index, band_count, page_count),
            )
        )
        typeset_ids.append(typeset_id)

    tasks.append(
        ScheduledTask(
            "reassemble",
            "reassemble",
            "cpu",
            depends_on=tuple(typeset_ids) if typeset_ids else ("translate_batch",),
        )
    )

    groups = build_execution_groups(tasks, max_cpu_parallel=max_cpu_parallel, max_gpu_parallel=max_gpu_parallel)
    validation = validate_schedule(tasks, groups, max_gpu_parallel=max_gpu_parallel)
    return StripSchedulerPlan(
        tasks=tasks,
        execution_groups=groups,
        max_cpu_parallel=max_cpu_parallel,
        max_gpu_parallel=max_gpu_parallel,
        validation=validation,
        notes=[
            "Contract only: run_chapter still executes bands sequentially.",
            "GPU tasks are serialized to protect 8GB-class cards from OCR/inpaint overlap.",
        ],
    )


def build_execution_groups(
    tasks: list[ScheduledTask],
    *,
    max_cpu_parallel: int,
    max_gpu_parallel: int,
) -> list[list[str]]:
    pending = {task.task_id: task for task in tasks}
    completed: set[str] = set()
    groups: list[list[str]] = []

    while pending:
        ready = [
            task
            for task in pending.values()
            if all(dep in completed for dep in task.depends_on)
        ]
        if not ready:
            raise ValueError("scheduler dependency cycle or missing dependency")

        ready.sort(key=_task_sort_key)
        cpu_slots = max_cpu_parallel
        gpu_slots = max_gpu_parallel
        group: list[ScheduledTask] = []
        for task in ready:
            if task.resource == "gpu":
                if gpu_slots <= 0:
                    continue
                gpu_slots -= 1
            else:
                if cpu_slots <= 0:
                    continue
                cpu_slots -= 1
            group.append(task)

        if not group:
            raise ValueError("scheduler could not place any ready task")
        group_ids = [task.task_id for task in group]
        groups.append(group_ids)
        for task_id in group_ids:
            pending.pop(task_id)
            completed.add(task_id)
    return groups


def validate_schedule(
    tasks: list[ScheduledTask],
    execution_groups: list[list[str]],
    *,
    max_gpu_parallel: int,
) -> ScheduleValidation:
    reasons: list[str] = []
    task_map = {task.task_id: task for task in tasks}
    if len(task_map) != len(tasks):
        reasons.append("duplicate task ids")

    seen: set[str] = set()
    for group_index, group in enumerate(execution_groups):
        gpu_count = 0
        for task_id in group:
            task = task_map.get(task_id)
            if task is None:
                reasons.append(f"group {group_index} references unknown task {task_id}")
                continue
            if task.resource == "gpu":
                gpu_count += 1
            missing_deps = [dep for dep in task.depends_on if dep not in seen]
            if missing_deps:
                reasons.append(f"task {task_id} scheduled before dependencies {missing_deps}")
        if gpu_count > max_gpu_parallel:
            reasons.append(
                f"GPU group exceeds single-worker limit at group {group_index}: {gpu_count}>{max_gpu_parallel}"
            )
        seen.update(group)

    missing_tasks = set(task_map) - seen
    if missing_tasks:
        reasons.append(f"tasks not scheduled: {sorted(missing_tasks)}")
    extra_tasks = seen - set(task_map)
    if extra_tasks:
        reasons.append(f"unknown tasks scheduled: {sorted(extra_tasks)}")

    if reasons:
        return ScheduleValidation("FAIL", reasons)
    return ScheduleValidation("PASS", ["schedule is topological and respects single GPU worker"])


def evaluate_scheduler_gate(
    out_dir: str | Path | None = None,
    *,
    band_count: int = 154,
    page_count: int = 27,
) -> dict[str, Any]:
    plan = build_strip_scheduler_plan(band_count=band_count, page_count=page_count)
    status = plan.validation.status
    result = {
        "gate": {
            "name": "strip_scheduler_contract",
            "status": status,
            "reasons": plan.validation.reasons,
            "task_count": plan.task_count,
            "cpu_task_count": plan.cpu_task_count,
            "gpu_task_count": plan.gpu_task_count,
            "stage_counts": plan.stage_counts,
            "max_cpu_parallel": plan.max_cpu_parallel,
            "max_gpu_parallel": plan.max_gpu_parallel,
            "notes": plan.notes,
        },
        "plan": plan.to_dict(),
    }
    return _write_result(result, out_dir)


def _resource_priority(task: ScheduledTask) -> int:
    return 0 if task.resource == "gpu" else 1


def _task_sort_key(task: ScheduledTask) -> tuple[int, int, int, int, str]:
    band_index = task.band if task.band is not None else -1
    page_index = task.page if task.page is not None else -1
    return (
        _resource_priority(task),
        _STAGE_PRIORITY.get(task.stage, 99),
        page_index,
        band_index,
        task.task_id,
    )


def _page_for_band(band_index: int, band_count: int, page_count: int) -> int:
    if band_count <= 0:
        return 1
    return min(page_count, int((band_index / band_count) * page_count) + 1)


def _write_result(result: dict[str, Any], out_dir: str | Path | None) -> dict[str, Any]:
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--band-count", type=int, default=154)
    parser.add_argument("--page-count", type=int, default=27)
    args = parser.parse_args(argv)

    result = evaluate_scheduler_gate(args.out, band_count=args.band_count, page_count=args.page_count)
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return 0 if result["gate"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
