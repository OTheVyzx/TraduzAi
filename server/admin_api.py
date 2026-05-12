from __future__ import annotations

from fastapi import APIRouter, Depends

from server.config import Settings
from server.db import session_scope
from server.deps import get_settings, require_admin
from server.models import AuditLog, Job, WorkerNode


router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/overview")
def overview(settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
        workers = db.query(WorkerNode).order_by(WorkerNode.updated_at.desc()).limit(20).all()
        audit_logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(50).all()
        return {
            "jobs": [
                {
                    "id": job.id,
                    "obra": job.obra,
                    "capitulo": job.capitulo,
                    "status": job.status,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }
                for job in jobs
            ],
            "workers": [
                {
                    "id": worker.id,
                    "name": worker.name,
                    "status": worker.status,
                    "max_concurrent_jobs": worker.max_concurrent_jobs,
                    "last_seen_at": worker.last_seen_at.isoformat() if worker.last_seen_at else None,
                }
                for worker in workers
            ],
            "audit_logs": [
                {
                    "id": log.id,
                    "action": log.action,
                    "entity_type": log.entity_type,
                    "entity_id": log.entity_id,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in audit_logs
            ],
        }
