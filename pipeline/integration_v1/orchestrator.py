"""Versioned Consumer Fast execution plan, invalidation and durable job state."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from ownership.hash_contract import canonical_json_sha256, sha256_text


STAGE_ORDER = (
    "import", "analysis", "ocr", "logical_units", "translate", "restore",
    "layout", "rasterize", "review", "persist", "export_decision",
)
CAPABILITY_ORDER = (
    "analysis", "continuity", "narration_multiline", "operational_recovery",
)


class RevisionConflict(RuntimeError):
    pass


def build_execution_plan(capabilities: dict[str, bool], *, policy_versions: dict[str, str]) -> dict[str, Any]:
    enabled = [name for name in CAPABILITY_ORDER if capabilities.get(name) is True]
    payload = {
        "schema": "traduzai.consumer-fast-plan.v1",
        "stages": list(STAGE_ORDER),
        "capabilities": enabled,
        "policy_versions": {key: policy_versions[key] for key in sorted(policy_versions)},
    }
    return {**payload, "fingerprint": canonical_json_sha256(payload)}


_INVALIDATION = {
    "typography": ("layout", "rasterize", "review", "persist", "export_decision"),
    "line_breaks": ("layout", "rasterize", "review", "persist", "export_decision"),
    "target": ("layout", "rasterize", "review", "persist", "export_decision"),
    "source_selection": (
        "ocr", "logical_units", "translate", "restore", "layout", "rasterize",
        "review", "persist", "export_decision",
    ),
    "mask": ("restore", "rasterize", "review", "persist", "export_decision"),
}


def invalidated_stages(change: str) -> tuple[str, ...]:
    try:
        return _INVALIDATION[str(change)]
    except KeyError as exc:
        raise ValueError(f"unsupported invalidation change: {change}") from exc


class ConsumerFastJobStore:
    """Small atomic store used by the Rust adapter and standalone runner."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "idempotency").mkdir(exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _idempotency_path(self, key: str) -> Path:
        return self.root / "idempotency" / f"{sha256_text(key)}.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.is_file():
            raise KeyError(f"unknown job: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _idempotent(self, key: str, operation) -> dict[str, Any]:
        receipt = self._idempotency_path(key)
        if receipt.is_file():
            return json.loads(receipt.read_text(encoding="utf-8"))["response"]
        response = operation()
        self._atomic_write(receipt, {"idempotency_key": key, "response": response})
        return response

    @staticmethod
    def _require_revision(state: dict[str, Any], expected_revision: int) -> None:
        actual = int(state["project_revision"])
        if actual != int(expected_revision):
            raise RevisionConflict(f"PROJECT_REVISION_CONFLICT expected={expected_revision} actual={actual}")

    def start(self, *, project_path: str, expected_revision: int,
              idempotency_key: str, unit_ids: Iterable[str]) -> dict[str, Any]:
        def operation():
            job_id = f"job-{sha256_text(f'{project_path}|{idempotency_key}')[:20]}"
            state = {
                "schema": "traduzai.consumer-fast-job.v1",
                "job_id": job_id,
                "project_path": str(project_path),
                "project_revision": int(expected_revision) + 1,
                "sequence": 1,
                "status": "running",
                "units": {str(unit): {"status": "queued", "terminal": False,
                                       "reason_code": ""} for unit in unit_ids},
            }
            self._atomic_write(self._job_path(job_id), state)
            return state
        return self._idempotent(idempotency_key, operation)

    def _mutate(self, job_id: str, expected_revision: int, change) -> dict[str, Any]:
        state = self._read(job_id)
        self._require_revision(state, expected_revision)
        change(state)
        state["project_revision"] = int(state["project_revision"]) + 1
        state["sequence"] = int(state["sequence"]) + 1
        self._atomic_write(self._job_path(job_id), state)
        return state

    def command(self, job_id: str, action: str, *, expected_revision: int,
                idempotency_key: str) -> dict[str, Any]:
        statuses = {"pause": "paused", "resume": "running", "cancel": "cancelled"}
        if action not in statuses:
            raise ValueError(f"unsupported job command: {action}")
        def operation():
            def change(state):
                if state["status"] in {"complete", "cancelled", "failed"}:
                    raise ValueError("terminal job cannot change state")
                state["status"] = statuses[action]
                if action == "cancel":
                    for unit in state["units"].values():
                        if not unit["terminal"]:
                            unit.update(status="cancelled", terminal=True,
                                        reason_code="job_cancelled")
            return self._mutate(job_id, expected_revision, change)
        return self._idempotent(idempotency_key, operation)

    def finish_unit(self, job_id: str, unit_id: str, status: str, *,
                    reason_code: str, expected_revision: int) -> dict[str, Any]:
        if status not in {"complete", "review_required", "preserved", "failed", "cancelled"}:
            raise ValueError("unit status is not terminal")
        def change(state):
            if unit_id not in state["units"]:
                raise KeyError(f"unknown unit: {unit_id}")
            state["units"][unit_id] = {
                "status": status, "terminal": True, "reason_code": str(reason_code),
            }
            if all(unit["terminal"] for unit in state["units"].values()):
                state["status"] = "complete_with_review" if any(
                    unit["status"] != "complete" for unit in state["units"].values()) else "complete"
        return self._mutate(job_id, expected_revision, change)

    def retry(self, job_id: str, unit_ids: Iterable[str], *, expected_revision: int,
              idempotency_key: str) -> dict[str, Any]:
        selected = tuple(str(unit) for unit in unit_ids)
        def operation():
            def change(state):
                for unit_id in selected:
                    unit = state["units"].get(unit_id)
                    if unit is None or not unit["terminal"]:
                        raise ValueError(f"unit is not retryable: {unit_id}")
                    state["units"][unit_id] = {"status": "queued", "terminal": False,
                                                "reason_code": "retry_requested"}
                state["status"] = "running"
            return self._mutate(job_id, expected_revision, change)
        return self._idempotent(idempotency_key, operation)
