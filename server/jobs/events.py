from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Job, JobEvent, User
from server.orgs import user_belongs_to_org


router = APIRouter(prefix="/api/jobs", tags=["events"])


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")

    start_id = int(last_event_id or "0")

    async def stream():
        current = start_id
        idle = 0
        while True:
            with session_scope(settings) as db:
                events = (
                    db.query(JobEvent)
                    .filter(JobEvent.job_id == job_id, JobEvent.id > current)
                    .order_by(JobEvent.id.asc())
                    .limit(100)
                    .all()
                )
            if events:
                idle = 0
                for event in events:
                    current = event.id
                    yield {
                        "id": str(event.id),
                        "event": event.kind,
                        "data": json.dumps(
                            {
                                "stage": event.stage,
                                "kind": event.kind,
                                "message": event.message,
                                "payload": json.loads(event.payload_json or "{}"),
                                "created_at": event.created_at.isoformat(),
                            },
                            ensure_ascii=True,
                        ),
                    }
            else:
                idle += 1
                yield {"event": "heartbeat", "data": "{}"}
                if idle > 360:
                    return
            await asyncio.sleep(2)

    return EventSourceResponse(stream())
