from __future__ import annotations

import json
from datetime import timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from server.audit import log_event
from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.jobs.uploads import prepare_upload, prepare_upload_from_drive_link
from server.models import Artifact, Job, JobEvent, User, new_id, utc_now
from server.orgs import default_org_for_user, user_belongs_to_org
from server.queue import enqueue
from server.storage import delete as storage_delete
from server.storage import put_file
from server.usage import record_usage_event
from server.vast.orchestrator import ensure_worker_available
from server.vast.serverless import ensure_serverless_job_started


router = APIRouter(prefix="/api/jobs", tags=["jobs"])
DEFAULT_CHAPTER = "1"
DEFAULT_WORK_TITLE = "Projeto sem nome"


def _utc_iso(value) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _job_payload(job: Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "obra": job.obra,
        "capitulo": job.capitulo,
        "src_lang": job.src_lang,
        "dst_lang": job.dst_lang,
        "mode": job.mode,
        "project_config": json.loads(job.config_json) if job.config_json else None,
        "page_count": job.page_count,
        "processing_seconds": job.processing_seconds,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": _utc_iso(job.created_at),
        "started_at": _utc_iso(job.started_at),
        "finished_at": _utc_iso(job.finished_at),
    }


@router.post("")
async def create_job(
    obra: str = Form(DEFAULT_WORK_TITLE),
    capitulo: str = Form(DEFAULT_CHAPTER),
    src_lang: str = Form("en"),
    dst_lang: str = Form("pt-BR"),
    mode: str = Form("mock"),
    project_config: str | None = Form(None),
    drive_link: str | None = Form(None),
    file: UploadFile | None = File(None),
    user: User = Depends(current_user),
    settings: Settings = Depends(get_settings),
):
    obra_value = obra.strip() or DEFAULT_WORK_TITLE
    capitulo_value = capitulo.strip() or DEFAULT_CHAPTER
    if file is not None:
        upload = await prepare_upload(file, settings)
    elif drive_link:
        upload = prepare_upload_from_drive_link(drive_link, settings)
    else:
        raise HTTPException(status_code=422, detail="selecione um arquivo ou informe um link do Google Drive")
    config_json = None
    config_data = None
    if project_config:
        try:
            config_data = json.loads(project_config)
            config_json = json.dumps(config_data, ensure_ascii=True)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="project_config invalido") from exc
    is_manual_project = isinstance(config_data, dict) and config_data.get("mode") == "manual"
    job_id = new_id()
    with session_scope(settings) as db:
        org = default_org_for_user(db, user.id)
        if org is None:
            raise HTTPException(status_code=403, detail="organizacao ausente")
        manual_finished_at = utc_now() if is_manual_project else None
        job = Job(
            id=job_id,
            organization_id=org.id,
            user_id=user.id,
            status="completed" if is_manual_project else "queued",
            obra=obra_value,
            capitulo=capitulo_value,
            src_lang=src_lang,
            dst_lang=dst_lang,
            mode=mode,
            config_json=config_json,
            page_count=upload.page_count,
            processing_seconds=0.0 if is_manual_project else None,
            started_at=manual_finished_at,
            finished_at=manual_finished_at,
        )
        db.add(job)
        db.flush()
        storage_key = f"jobs/{job_id}/input/{new_id()}{upload.suffix}"
        put_file(upload.path, storage_key, upload.mime_type, settings)
        db.add(
            Artifact(
                job_id=job_id,
                organization_id=org.id,
                kind="input_archive" if upload.suffix in {".zip", ".cbz"} else "input_original",
                storage_key=storage_key,
                filename=upload.filename,
                mime_type=upload.mime_type,
                size=upload.size,
                sha256=upload.sha256,
            )
        )
        db.add(
            JobEvent(
                job_id=job_id,
                organization_id=org.id,
                stage="manual" if is_manual_project else "queue",
                kind="status",
                message="Projeto manual pronto para edicao" if is_manual_project else "Job enfileirado",
            )
        )
        log_event(db, action="job.created", entity_type="job", entity_id=job_id, organization_id=org.id, user_id=user.id)
    upload.path.unlink(missing_ok=True)
    if not is_manual_project:
        enqueue(settings, job_id)
        _ensure_worker_available_safely(settings, job_id)
    return {"job": {"id": job_id, "status": "completed" if is_manual_project else "queued"}}


@router.get("")
def list_jobs(user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        org = default_org_for_user(db, user.id)
        if org is None:
            return {"jobs": []}
        jobs = (
            db.query(Job)
            .filter(Job.organization_id == org.id, Job.status != "deleted")
            .order_by(Job.created_at.desc())
            .limit(100)
            .all()
        )
        return {"jobs": [_job_payload(job) for job in jobs]}


@router.get("/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        artifacts = db.query(Artifact).filter_by(job_id=job_id).all()
        payload = _job_payload(job)
        payload["artifacts"] = [
            {
                "id": item.id,
                "kind": item.kind,
                "filename": item.filename,
                "size": item.size,
                "download_url": f"/api/artifacts/{item.id}",
            }
            for item in artifacts
        ]
        return {"job": payload}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        if job.status in {"completed", "failed", "cancelled"}:
            return {"job": _job_payload(job)}
        job.status = "cancelled"
        job.cancel_requested_at = utc_now()
        job.finished_at = utc_now()
        record_usage_event(db, job, "job_cancelled")
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, stage="queue", kind="status", message="Job cancelado"))
        log_event(db, action="job.cancelled", entity_type="job", entity_id=job.id, organization_id=job.organization_id, user_id=user.id)
        return {"job": _job_payload(job)}


@router.post("/{job_id}/retry")
def retry_job(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        if job.status not in {"failed", "cancelled"}:
            raise HTTPException(status_code=409, detail="apenas jobs falhados ou cancelados podem ser reenfileirados")
        input_artifact = (
            db.query(Artifact)
            .filter(Artifact.job_id == job.id, Artifact.kind.in_(["input_original", "input_archive"]))
            .order_by(Artifact.created_at.asc())
            .first()
        )
        if input_artifact is None:
            raise HTTPException(status_code=409, detail="job sem arquivo de entrada")
        stale_artifacts = (
            db.query(Artifact)
            .filter(Artifact.job_id == job.id, ~Artifact.kind.in_(["input_original", "input_archive"]))
            .all()
        )
        for artifact in stale_artifacts:
            storage_delete(artifact.storage_key, settings)
            db.delete(artifact)
        job.status = "queued"
        job.worker_id = None
        job.error_code = None
        job.error_message = None
        job.processing_seconds = None
        job.claimed_at = None
        job.claimed_until = None
        job.started_at = None
        job.last_heartbeat_at = None
        job.finished_at = None
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, stage="queue", kind="status", message="Job reenfileirado"))
        log_event(db, action="job.retry", entity_type="job", entity_id=job.id, organization_id=job.organization_id, user_id=user.id)
        payload = _job_payload(job)
    _ensure_worker_available_safely(settings, job_id)
    return {"job": payload}


@router.delete("/{job_id}")
def delete_job(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        artifacts = db.query(Artifact).filter_by(job_id=job_id).all()
        for artifact in artifacts:
            storage_delete(artifact.storage_key, settings)
        job.status = "deleted"
        job.deleted_at = utc_now()
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, stage="storage", kind="status", message="Job excluido"))
        log_event(db, action="job.deleted", entity_type="job", entity_id=job.id, organization_id=job.organization_id, user_id=user.id)
        return {"ok": True}


def _ensure_worker_available_safely(settings: Settings, job_id: str) -> None:
    if not settings.vast_autostart:
        return
    try:
        if settings.vast_provider == "serverless":
            result = ensure_serverless_job_started(settings, job_id)
        else:
            result = ensure_worker_available(settings)
        message = f"Vast auto-start: {result.get('action')}"
        kind = "status" if result.get("ok") else "warning"
    except Exception as exc:
        message = f"Vast auto-start falhou: {exc}"
        kind = "warning"
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        db.add(JobEvent(job_id=job.id, organization_id=job.organization_id, stage="queue", kind=kind, message=message))
