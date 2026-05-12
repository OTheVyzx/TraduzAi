from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Artifact, Job, User, new_id
from server.projects.api import _require_project_access
from server.projects.workspace import IMAGE_SUFFIXES, load_project, project_root
from server.storage import put_file


router = APIRouter(prefix="/api/projects/{project_id}/exports", tags=["exports"])


@router.post("/zip-full")
def export_zip_full(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    output = root / "exports" / f"traduzai-{project_id}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"project_id": project_id, "format": "zip-full", "includes": ["project.json", "translated", "originals", "layers"]}
    (root / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".bundle-source.zip" in path.name or path == output:
                continue
            archive.write(path, path.relative_to(root).as_posix())
    return {"artifact": _register_export(project_id, output, "bundle_zip", "application/zip", user, settings)}


@router.post("/cbz")
def export_cbz(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    output = root / "exports" / f"traduzai-{project_id}.cbz"
    output.parent.mkdir(parents=True, exist_ok=True)
    images = _final_images(root)
    if not images:
        raise HTTPException(status_code=404, detail="nenhuma pagina final encontrada")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images, 1):
            archive.write(image, f"{index:03d}.jpg")
    return {"artifact": _register_export(project_id, output, "bundle_zip", "application/vnd.comicbook+zip", user, settings)}


@router.post("/jpg-zip")
def export_jpg_zip(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    output = root / "exports" / f"traduzai-{project_id}-jpg.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    images = _final_images(root)
    if not images:
        raise HTTPException(status_code=404, detail="nenhuma pagina final encontrada")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, image in enumerate(images, 1):
            archive.write(image, f"translated/{index:03d}.jpg")
    return {"artifact": _register_export(project_id, output, "bundle_zip", "application/zip", user, settings)}


@router.post("/psd-page")
def export_psd_page(payload: dict, project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    page_index = int(payload.get("page_index", 0))
    project = load_project(project_root(project_id, settings))
    if page_index < 0 or page_index >= len(project.get("paginas") or []):
        raise HTTPException(status_code=404, detail="pagina nao encontrada")
    root = project_root(project_id, settings)
    output = root / "exports" / f"page-{page_index + 1:03d}.psd"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"8BPS\x00\x01" + b"\x00" * 24)
    return {"artifact": _register_export(project_id, output, "export_manifest", "image/vnd.adobe.photoshop", user, settings)}


def _final_images(root: Path) -> list[Path]:
    translated = root / "translated"
    images = [path for path in translated.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        project = load_project(root)
        for page in project.get("paginas") or []:
            rel = page.get("rendered_path") or page.get("translated_path")
            if rel:
                path = root / rel
                if path.exists():
                    images.append(path)
    return sorted(images)


def _register_export(project_id: str, path: Path, kind: str, mime_type: str, user: User, settings: Settings) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with session_scope(settings) as db:
        job = db.get(Job, project_id)
        if job is None:
            raise HTTPException(status_code=404, detail="projeto nao encontrado")
        storage_key = f"jobs/{project_id}/exports/{new_id()}-{path.name}"
        put_file(path, storage_key, mime_type, settings)
        artifact = Artifact(
            job_id=project_id,
            organization_id=job.organization_id,
            kind=kind,
            storage_key=storage_key,
            filename=path.name,
            mime_type=mime_type,
            size=path.stat().st_size,
            sha256=digest,
        )
        db.add(artifact)
        db.flush()
        return {"id": artifact.id, "kind": artifact.kind, "filename": artifact.filename, "size": artifact.size, "download_url": f"/api/artifacts/{artifact.id}"}
