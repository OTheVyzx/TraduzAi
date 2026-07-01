from __future__ import annotations

import hashlib
import json
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.audit import log_event
from server.config import Settings
from server.db import session_scope
from server.deps import get_settings
from server.models import Artifact, Job, JobEvent, WorkerHeartbeat, WorkerNode, new_id, utc_now
from server.queue import claim_next, claim_specific
from server.storage import delete as delete_storage
from server.storage import open_for_read
from server.storage import put_file
from server.usage import record_usage_event
from server.workers.auth import require_worker_token


router = APIRouter(prefix="/api/workers", tags=["workers"], dependencies=[Depends(require_worker_token)])

KNOWN_ARTIFACT_KINDS = {
    "input_archive",
    "input_original",
    "project_json",
    "translated_image",
    "original_image",
    "inpaint_image",
    "layer_mask",
    "layer_brush",
    "layer_recovery",
    "preview_image",
    "bundle_zip",
    "pipeline_log",
    "runner_log",
    "export_manifest",
}


class RegisterPayload(BaseModel):
    name: str
    capabilities: dict = {}
    max_concurrent_jobs: int = 1


class HeartbeatPayload(BaseModel):
    worker_id: str
    status: str = "online"
    payload: dict = {}


class ClaimPayload(BaseModel):
    worker_id: str
    capabilities: dict = {}
    job_id: str | None = None


class EventPayload(BaseModel):
    worker_id: str
    stage: str
    kind: str = "status"
    message: str
    payload: dict = {}


class CompletePayload(BaseModel):
    worker_id: str
    page_count: int = 0
    processing_seconds: float = 0.0


class FailPayload(BaseModel):
    worker_id: str
    error_code: str
    error_message: str


@router.post("/register")
def register(payload: RegisterPayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        worker = db.query(WorkerNode).filter_by(name=payload.name).one_or_none()
        if worker is None:
            worker = WorkerNode(id=new_id(), name=payload.name)
            db.add(worker)
            db.flush()
        worker.status = "online"
        worker.capabilities_json = json.dumps(payload.capabilities, ensure_ascii=True)
        worker.max_concurrent_jobs = payload.max_concurrent_jobs
        worker.last_seen_at = utc_now()
        log_event(db, action="worker.register", entity_type="worker", entity_id=worker.id, worker_id=worker.id)
        return {"worker_id": worker.id}


@router.post("/heartbeat")
def heartbeat(payload: HeartbeatPayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        worker = db.get(WorkerNode, payload.worker_id)
        if worker is None:
            raise HTTPException(status_code=404, detail="worker nao encontrado")
        worker.status = payload.status
        worker.last_seen_at = utc_now()
        db.add(
            WorkerHeartbeat(
                worker_id=worker.id,
                status=payload.status,
                payload_json=json.dumps(payload.payload, ensure_ascii=True),
            )
        )
        for job in db.query(Job).filter(Job.worker_id == worker.id, Job.status.in_(["claimed", "running", "uploading_results"])):
            job.last_heartbeat_at = utc_now()
        return {"ok": True}


@router.post("/claim-job")
def claim_job(payload: ClaimPayload, settings: Settings = Depends(get_settings)):
    if payload.job_id:
        job = claim_specific(settings, payload.worker_id, payload.job_id, payload.capabilities)
    else:
        job = claim_next(settings, payload.worker_id, payload.capabilities)
    if job is None:
        return {"job": None}
    with session_scope(settings) as db:
        input_artifact = (
            db.query(Artifact)
            .filter(Artifact.job_id == job.id, Artifact.kind.in_(["input_original", "input_archive"]))
            .order_by(Artifact.created_at.asc())
            .first()
        )
    return {
        "job": {
            "id": job.id,
            "obra": job.obra,
            "capitulo": job.capitulo,
            "src_lang": job.src_lang,
            "dst_lang": job.dst_lang,
            "mode": job.mode,
            "project_config": json.loads(job.config_json) if job.config_json else None,
            "input_filename": input_artifact.filename if input_artifact else None,
            "input_download_url": f"/api/workers/jobs/{job.id}/input" if input_artifact else None,
        }
    }


@router.get("/jobs/{job_id}/input")
def download_input(job_id: str, worker_id: str, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, worker_id)
        artifact = (
            db.query(Artifact)
            .filter(Artifact.job_id == job.id, Artifact.kind.in_(["input_original", "input_archive"]))
            .order_by(Artifact.created_at.asc())
            .first()
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail="input nao encontrado")
        with open_for_read(artifact.storage_key, settings) as handle:
            data = handle.read()
        headers = {"Content-Disposition": f'attachment; filename="{artifact.filename}"'}
        return StreamingResponse(BytesIO(data), media_type=artifact.mime_type, headers=headers)


@router.post("/jobs/{job_id}/event")
def post_event(job_id: str, payload: EventPayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, payload.worker_id)
        if job.status == "claimed":
            job.status = "running"
            job.started_at = utc_now()
        job.last_heartbeat_at = utc_now()
        db.add(
            JobEvent(
                job_id=job.id,
                organization_id=job.organization_id,
                worker_id=payload.worker_id,
                stage=payload.stage,
                kind=payload.kind,
                message=payload.message,
                payload_json=json.dumps(payload.payload, ensure_ascii=True),
            )
        )
        return {"ok": True}


@router.post("/jobs/{job_id}/artifact")
async def post_artifact(
    job_id: str,
    worker_id: str = Form(...),
    kind: str = Form(...),
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    if kind not in KNOWN_ARTIFACT_KINDS:
        raise HTTPException(status_code=422, detail="tipo de artifact invalido")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = Path(tmp.name)
    digest = hashlib.sha256()
    size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            tmp.write(chunk)
    finally:
        tmp.close()
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, worker_id)
        job.status = "uploading_results"
        job.last_heartbeat_at = utc_now()
        storage_key = f"jobs/{job_id}/artifacts/{new_id()}-{Path(file.filename or 'artifact.bin').name}"
        put_file(tmp_path, storage_key, file.content_type, settings)
        artifact = Artifact(
            job_id=job.id,
            organization_id=job.organization_id,
            kind=kind,
            storage_key=storage_key,
            filename=Path(file.filename or "artifact.bin").name,
            mime_type=file.content_type,
            size=size,
            sha256=digest.hexdigest(),
        )
        db.add(artifact)
        existing_bundle = db.query(Artifact).filter_by(job_id=job.id, kind="bundle_zip").one_or_none()
        if existing_bundle is not None:
            delete_storage(existing_bundle.storage_key, settings)
            db.delete(existing_bundle)
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, worker_id=worker_id, stage="artifact", kind="artifact", message=f"Artifact {kind} enviado"))
    tmp_path.unlink(missing_ok=True)
    return {"ok": True}


@router.post("/jobs/{job_id}/complete")
def complete(job_id: str, payload: CompletePayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, payload.worker_id)
        job.status = "completed"
        job.page_count = payload.page_count
        job.processing_seconds = payload.processing_seconds
        job.finished_at = utc_now()
        job.last_heartbeat_at = utc_now()
        record_usage_event(db, job, "job_completed")
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, worker_id=payload.worker_id, stage="done", kind="status", message="Job concluido"))
        log_event(db, action="job.completed", entity_type="job", entity_id=job.id, organization_id=job.organization_id, worker_id=payload.worker_id)
        return {"ok": True}


@router.post("/jobs/{job_id}/fail")
def fail(job_id: str, payload: FailPayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, payload.worker_id)
        job.status = "failed"
        job.error_code = payload.error_code
        job.error_message = payload.error_message
        job.finished_at = utc_now()
        record_usage_event(db, job, "job_failed")
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, worker_id=payload.worker_id, stage="error", kind="error", message=payload.error_message))
        log_event(db, action="job.failed", entity_type="job", entity_id=job.id, organization_id=job.organization_id, worker_id=payload.worker_id)
        return {"ok": True}


@router.post("/jobs/{job_id}/cancelled")
def cancelled(job_id: str, payload: CompletePayload, settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = _worker_job(db, job_id, payload.worker_id)
        job.status = "cancelled"
        job.finished_at = utc_now()
        record_usage_event(db, job, "job_cancelled")
        return {"ok": True}


def _worker_job(db, job_id: str, worker_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.worker_id != worker_id:
        raise HTTPException(status_code=404, detail="job nao encontrado para worker")
    return job
