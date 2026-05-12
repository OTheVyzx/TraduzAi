from __future__ import annotations

import json

from server.models import Job, UsageEvent


def record_usage_event(db, job: Job, event_type: str, metadata: dict | None = None) -> None:
    pages = int(job.page_count or 0)
    db.add(
        UsageEvent(
            organization_id=job.organization_id,
            job_id=job.id,
            event_type=event_type,
            pages=pages,
            processing_seconds=job.processing_seconds,
            estimated_credits=pages,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=True),
        )
    )
