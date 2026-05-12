from __future__ import annotations

from datetime import timedelta

from server.audit import log_event
from server.config import Settings
from server.db import session_scope
from server.models import Job, utc_now
from server.usage import record_usage_event


def _older_than(value, cutoff) -> bool:
    if value is None:
        return True
    if value.tzinfo is None:
        value = value.replace(tzinfo=cutoff.tzinfo)
    return value < cutoff


def fail_lost_jobs(settings: Settings) -> int:
    cutoff = utc_now() - timedelta(seconds=settings.lease_timeout_seconds)
    changed = 0
    with session_scope(settings) as session:
        jobs = session.query(Job).filter(Job.status.in_(["claimed", "running", "uploading_results"])).all()
        for job in jobs:
            if not _older_than(job.last_heartbeat_at, cutoff):
                continue
            job.status = "failed"
            job.error_code = "worker_lost"
            job.error_message = "heartbeat ausente"
            job.finished_at = utc_now()
            record_usage_event(session, job, "job_failed", {"reason": "worker_lost"})
            log_event(
                session,
                action="job.worker_lost",
                entity_type="job",
                entity_id=job.id,
                organization_id=job.organization_id,
                worker_id=job.worker_id,
            )
            changed += 1
    return changed
