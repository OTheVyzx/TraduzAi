from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Artifact, Job, User, new_id
from server.orgs import default_org_for_user, user_belongs_to_org
from server.projects.workspace import (
    load_project,
    load_state,
    materialize_workspace,
    page_at,
    project_root,
    relative_asset,
    safe_path,
    save_project,
    save_state,
)
from server.projects.pipeline_runner import render_preview_page
from server.storage import put_file


router = APIRouter(tags=["projects"])


@router.post("/api/jobs/{job_id}/materialize-project")
def materialize_project(job_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    with session_scope(settings) as db:
        job = db.get(Job, job_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="job nao encontrado")
        artifacts = db.query(Artifact).filter_by(job_id=job_id).all()
        return materialize_workspace(job, artifacts, settings)


@router.post("/api/projects/import")
async def import_project(file: UploadFile = File(...), user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=422, detail="envie um ZIP completo")
    project_id = new_id()
    root = project_root(project_id, settings)
    temp = root / "import.zip"
    temp.write_bytes(await file.read())
    try:
        with zipfile.ZipFile(temp) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                target = safe_path(root, item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source:
                    target.write_bytes(source.read())
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="zip invalido") from exc
    finally:
        temp.unlink(missing_ok=True)
    project = load_project(root)
    with session_scope(settings) as db:
        org = default_org_for_user(db, user.id)
        if org is None:
            raise HTTPException(status_code=403, detail="organizacao ausente")
        job = Job(
            id=project_id,
            organization_id=org.id,
            user_id=user.id,
            status="completed",
            obra=project.get("obra") or "Projeto importado",
            capitulo=str(project.get("capitulo") or "1"),
            src_lang=project.get("idioma_origem") or "en",
            dst_lang=project.get("idioma_destino") or "pt-BR",
            mode="import",
            page_count=len(project.get("paginas") or []),
        )
        db.add(job)
        db.flush()
        put_file(root / "project.json", f"jobs/{project_id}/project/project.json", "application/json", settings)
        db.add(
            Artifact(
                job_id=project_id,
                organization_id=org.id,
                kind="project_json",
                storage_key=f"jobs/{project_id}/project/project.json",
                filename="project.json",
                mime_type="application/json",
                size=(root / "project.json").stat().st_size,
                sha256=hashlib.sha256((root / "project.json").read_bytes()).hexdigest(),
            )
        )
    return {"project_id": project_id, "page_count": len(project.get("paginas") or [])}


@router.get("/api/projects/{project_id}")
def get_project(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    state = load_state(root)
    return {"project": project, "state": state}


@router.put("/api/projects/{project_id}")
def put_project(project_id: str, project: dict, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    save_project(root, project)
    return {"ok": True}


@router.get("/api/projects/{project_id}/pages/{page_index}")
def get_page(project_id: str, page_index: int, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    return {"page": page, "layers": _page_layers(root, page, page_index), "state": load_state(root)}


@router.get("/api/projects/{project_id}/assets/{asset_path:path}")
def get_asset(project_id: str, asset_path: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    path = safe_path(root, asset_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="asset nao encontrado")
    return FileResponse(path)


@router.post("/api/projects/{project_id}/pages/{page_index}/render-preview")
def render_preview(project_id: str, page_index: int, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    page = page_at(project, page_index)
    asset_path = render_preview_page(root, page_index, page)
    state = load_state(root)
    preview = state.setdefault("preview", {})
    preview[str(page_index)] = {"status": "fresh", "asset_path": asset_path}
    state["dirty"] = False
    save_state(root, state)
    return {"preview_url": f"/api/projects/{project_id}/assets/{asset_path}", "asset_path": asset_path}


@router.get("/api/projects/{project_id}/settings")
def get_project_settings(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    project = load_project(project_root(project_id, settings))
    return {"settings": project.get("config") or {}, "project": {key: project.get(key) for key in ["obra", "capitulo", "idioma_origem", "idioma_destino"]}}


@router.put("/api/projects/{project_id}/settings")
def put_project_settings(project_id: str, payload: dict, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    project = load_project(root)
    for key in ["obra", "capitulo", "idioma_origem", "idioma_destino"]:
        if key in payload:
            project[key] = payload[key]
    if "config" in payload:
        project["config"] = payload["config"]
    save_project(root, project)
    return {"ok": True}


def _require_project_access(project_id: str, user: User, settings: Settings) -> Job:
    with session_scope(settings) as db:
        job = db.get(Job, project_id)
        if job is None or not user_belongs_to_org(db, user.id, job.organization_id):
            raise HTTPException(status_code=404, detail="projeto nao encontrado")
        return job


def _asset_url(project_id: str, rel: str) -> str:
    return f"/api/projects/{project_id}/assets/{rel}"


def _page_asset_path(page: dict, key: str) -> str | None:
    legacy_map = {
        "base": "original_path",
        "rendered": "rendered_path",
        "translated": "translated_path",
        "inpaint": "inpaint_path",
    }
    legacy = legacy_map.get(key)
    if legacy and page.get(legacy):
        return str(page[legacy]).replace("\\", "/")
    image_layers = page.get("image_layers") or {}
    layer = image_layers.get(key) or {}
    rel = layer.get("path")
    if rel:
        return str(rel).replace("\\", "/")
    return None


def _page_layers(root: Path, page: dict, page_index: int) -> dict:
    project_id = root.name
    layers = {}
    candidates = {
        "base": _page_asset_path(page, "base"),
        "rendered": _page_asset_path(page, "rendered") or _page_asset_path(page, "translated"),
        "translated": _page_asset_path(page, "translated"),
        "inpaint": _page_asset_path(page, "inpaint"),
        "mask": f"layers/mask/{page_index + 1:03d}.png",
        "brush": f"layers/brush/{page_index + 1:03d}.png",
        "recovery": f"layers/recovery/{page_index + 1:03d}.png",
    }
    for key, rel in candidates.items():
        if not rel:
            continue
        try:
            path = safe_path(root, rel)
        except HTTPException:
            continue
        if path.exists():
            layers[key] = {"asset_path": rel, "url": _asset_url(project_id, rel)}
    return layers
