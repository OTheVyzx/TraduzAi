from __future__ import annotations

from datetime import timedelta

from server.config import Settings
from server.db import session_scope
from server.models import Job, WorkerNode, utc_now


BUSY_STATUSES = ["claimed", "running", "uploading_results"]


def enqueue(settings: Settings, job_id: str) -> None:
    with session_scope(settings) as session:
        job = session.get(Job, job_id)
        if job is None:
            raise ValueError("job inexistente")
        if job.status != "queued":
            job.status = "queued"
            job.worker_id = None
            job.error_code = None
            job.error_message = None


def claim_next(settings: Settings, worker_id: str, capabilities: dict) -> Job | None:
    with session_scope(settings) as session:
        worker = session.get(WorkerNode, worker_id)
        if worker is None:
            return None
        running_count = session.query(Job).filter(Job.worker_id == worker_id, Job.status.in_(BUSY_STATUSES)).count()
        if running_count >= worker.max_concurrent_jobs:
            return None

        # Postgres: SELECT ... FOR UPDATE SKIP LOCKED
        query = session.query(Job).filter_by(status="queued")
        modes = capabilities.get("mode") if isinstance(capabilities, dict) else None
        if modes:
            query = query.filter(Job.mode.in_(list(modes)))
        candidate = query.order_by(Job.created_at.asc()).first()
        if candidate is None:
            return None
        updated = (
            session.query(Job)
            .filter(Job.id == candidate.id, Job.status == "queued")
            .update(
                {
                    "status": "claimed",
                    "worker_id": worker_id,
                    "claimed_at": utc_now(),
                    "claimed_until": utc_now() + timedelta(seconds=settings.lease_timeout_seconds),
                    "last_heartbeat_at": utc_now(),
                },
                synchronize_session=False,
            )
        )
        if updated != 1:
            return None
        session.flush()
        return session.get(Job, candidate.id)


def claim_specific(settings: Settings, worker_id: str, job_id: str, capabilities: dict) -> Job | None:
    with session_scope(settings) as session:
        worker = session.get(WorkerNode, worker_id)
        if worker is None:
            return None
        running_count = session.query(Job).filter(Job.worker_id == worker_id, Job.status.in_(BUSY_STATUSES)).count()
        if running_count >= worker.max_concurrent_jobs:
            return None

        query = session.query(Job).filter(Job.id == job_id, Job.status == "queued")
        modes = capabilities.get("mode") if isinstance(capabilities, dict) else None
        if modes:
            query = query.filter(Job.mode.in_(list(modes)))
        candidate = query.first()
        if candidate is None:
            return None
        updated = (
            session.query(Job)
            .filter(Job.id == candidate.id, Job.status == "queued")
            .update(
                {
                    "status": "claimed",
                    "worker_id": worker_id,
                    "claimed_at": utc_now(),
                    "claimed_until": utc_now() + timedelta(seconds=settings.lease_timeout_seconds),
                    "last_heartbeat_at": utc_now(),
                },
                synchronize_session=False,
            )
        )
        if updated != 1:
            return None
        session.flush()
        return session.get(Job, candidate.id)


def release(settings: Settings, job_id: str, reason: str) -> None:
    with session_scope(settings) as session:
        job = session.get(Job, job_id)
        if job is None:
            raise ValueError("job inexistente")
        job.status = "failed"
        job.error_code = reason
        job.worker_id = None
        job.finished_at = utc_now()
