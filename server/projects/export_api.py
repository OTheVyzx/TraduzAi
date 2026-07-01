from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Artifact, Job, User, new_id
from server.projects.api import _require_project_access
from server.projects.psd_export import export_project_page_psd
from server.projects.workspace import IMAGE_SUFFIXES, load_project, project_root
from server.storage import put_file


router = APIRouter(prefix="/api/projects/{project_id}/exports", tags=["exports"])


@router.post("/zip-full")
def export_zip_full(project_id: str, user: User = Depends(current_user), settings: Settings = Depends(get_settings)):
    _require_project_access(project_id, user, settings)
    root = project_root(project_id, settings)
    output = root / "exports" / f"traduzai-{project_id}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, str]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ["translated", "originals", "images", "layers"]:
            _add_directory_to_zip(archive, root / folder, f"{folder}/", manifest_files)
        project_json = root / "project.json"
        if project_json.exists():
            _add_file_to_zip(archive, project_json, "project.json", manifest_files)
        _add_quality_files(archive, root, manifest_files)
        manifest = {
            "run_id": project_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "with_warnings",
            "files": manifest_files,
        }
        archive.writestr("export_manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))
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
        _add_final_images_flat(archive, images)
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
        _add_final_images_flat(archive, images)
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
    output.write_bytes(export_project_page_psd(root, project, page_index))
    return {"artifact": _register_export(project_id, output, "export_manifest", "image/vnd.adobe.photoshop", user, settings)}


def _final_images(root: Path) -> list[Path]:
    translated = root / "translated"
    images = [path for path in translated.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        project = load_project(root)
        for page in project.get("paginas") or []:
            rel = _page_final_image_rel(page)
            if rel:
                path = root / rel
                if path.exists():
                    images.append(path)
    return sorted(images)


def _page_final_image_rel(page: dict) -> str | None:
    for key in ["arquivo_traduzido", "rendered_path", "translated_path"]:
        rel = page.get(key)
        if rel:
            return str(rel).replace("\\", "/")
    image_layers = page.get("image_layers") if isinstance(page.get("image_layers"), dict) else {}
    for key in ["rendered", "inpaint", "base"]:
        layer = image_layers.get(key) if isinstance(image_layers.get(key), dict) else {}
        rel = layer.get("path")
        if rel:
            return str(rel).replace("\\", "/")
    return None


def _add_file_to_zip(archive: zipfile.ZipFile, path: Path, arcname: str, manifest_files: list[dict[str, str]]) -> None:
    payload = path.read_bytes()
    archive.writestr(arcname, payload)
    manifest_files.append({"path": arcname, "sha256": hashlib.sha256(payload).hexdigest()})


def _add_directory_to_zip(
    archive: zipfile.ZipFile,
    directory: Path,
    prefix: str,
    manifest_files: list[dict[str, str]],
) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        arcname = f"{prefix}{path.relative_to(directory).as_posix()}"
        _add_file_to_zip(archive, path, arcname, manifest_files)


def _add_quality_files(archive: zipfile.ZipFile, root: Path, manifest_files: list[dict[str, str]]) -> None:
    structured_log = root / "structured_log.jsonl"
    quality_files = {
        "qa_report.md": b"# Relatorio de QA\n\nExportado pelo site.\n",
        "qa_report.json": json.dumps({"summary": {"status": "with_warnings"}, "issues": [], "user_actions": []}, ensure_ascii=True, indent=2).encode("utf-8"),
        "issues.csv": b"id,page,severity,kind,message\n",
        "glossary_used.json": b"[]",
        "ocr_corrections.json": b"[]",
        "structured_log.jsonl": structured_log.read_bytes() if structured_log.exists() else b'{"event":"export","status":"generated"}\n',
    }
    for arcname, payload in quality_files.items():
        archive.writestr(arcname, payload)
        manifest_files.append({"path": arcname, "sha256": hashlib.sha256(payload).hexdigest()})


def _add_final_images_flat(archive: zipfile.ZipFile, images: list[Path]) -> None:
    used_names: set[str] = set()
    for index, image in enumerate(images, 1):
        arcname = image.name or f"{index:03d}{image.suffix.lower()}"
        if arcname in used_names:
            arcname = f"{index:03d}{image.suffix.lower()}"
        used_names.add(arcname)
        archive.write(image, arcname)


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
