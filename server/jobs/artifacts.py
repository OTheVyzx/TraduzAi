from __future__ import annotations

import tempfile
import zipfile
from io import BytesIO
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Artifact, Job, User, new_id
from server.orgs import user_belongs_to_org
from server.storage import open_for_read, put_file


router = APIRouter(prefix="/api", tags=["artifacts"])


@router.get("/artifacts/{artifact_id}")
def download_artifact(artifact_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        artifact = db.get(Artifact, artifact_id)
        if artifact is None or not user_belongs_to_org(db, user.id, artifact.organization_id):
            raise HTTPException(status_code=404, detail="artifact nao encontrado")
        headers = {"Content-Disposition": f'attachment; filename="{artifact.filename}"'}
        with open_for_read(artifact.storage_key, settings) as handle:
            data = handle.read()
        return StreamingResponse(BytesIO(data), media_type=artifact.mime_type, headers=headers)


@router.get("/jobs/{job_id}/download/zip")
def download_job_zip(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        existing = db.query(Artifact).filter_by(job_id=job_id, kind="bundle_zip").one_or_none()
        if existing is None:
            artifacts = db.query(Artifact).filter(Artifact.job_id == job_id, Artifact.kind != "bundle_zip").all()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp.close()
            with zipfile.ZipFile(tmp.name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for artifact in artifacts:
                    with open_for_read(artifact.storage_key, settings) as handle:
                        archive.writestr(_zip_member_name(artifact), handle.read())
            storage_key = f"jobs/{job_id}/bundle/{new_id()}.zip"
            from pathlib import Path
            import hashlib

            tmp_path = Path(tmp.name)
            data = tmp_path.read_bytes()
            put_file(tmp_path, storage_key, "application/zip", settings)
            existing = Artifact(
                job_id=job_id,
                organization_id=job.organization_id,
                kind="bundle_zip",
                storage_key=storage_key,
                filename=f"traduzai-{job_id}.zip",
                mime_type="application/zip",
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            db.add(existing)
            db.flush()
            tmp_path.unlink(missing_ok=True)
        headers = {"Content-Disposition": f'attachment; filename="{existing.filename}"'}
        with open_for_read(existing.storage_key, settings) as handle:
            data = handle.read()
        return StreamingResponse(BytesIO(data), media_type="application/zip", headers=headers)


def _zip_member_name(artifact: Artifact) -> str:
    raw = artifact.filename.replace("\\", "/")
    filename = "/".join(part for part in PurePosixPath(raw).parts if part not in {"", ".", "..", "/"})
    if artifact.kind == "translated_image":
        return f"translated/{filename}"
    return filename
