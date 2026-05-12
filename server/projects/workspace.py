from __future__ import annotations

import base64
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import HTTPException

from server.config import Settings
from server.models import Artifact, Job
from server.storage import get_file, root_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TRANSPARENT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def project_root(job_id: str, settings: Settings) -> Path:
    root = (root_path(settings) / "projects" / job_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_project_dirs(root: Path) -> None:
    for rel in [
        "originals",
        "images",
        "translated",
        "layers/mask",
        "layers/brush",
        "layers/recovery",
        "layers/text-preview",
        "render-cache/preview",
        "exports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def materialize_workspace(job: Job, artifacts: list[Artifact], settings: Settings) -> dict[str, Any]:
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="job ainda nao concluido")
    root = project_root(job.id, settings)
    ensure_project_dirs(root)
    bundle = next((item for item in artifacts if item.kind == "bundle_zip"), None)
    if bundle is not None:
        _extract_bundle(bundle, root, settings)
    else:
        input_archive = next((item for item in artifacts if item.kind == "input_archive"), None)
        if input_archive is not None:
            _extract_input_archive(input_archive, root, settings)
        _copy_artifacts(artifacts, root, settings)
    project_path = root / "project.json"
    if not project_path.exists():
        project_path.write_text(json.dumps(_build_minimal_project(job, root), ensure_ascii=True, indent=2), encoding="utf-8")
    project = load_project(root)
    state = {
        "job_id": job.id,
        "source_artifact_ids": [item.id for item in artifacts],
        "materialized_at": datetime.now(timezone.utc).isoformat(),
        "dirty": False,
        "preview": {},
    }
    save_state(root, state)
    return {"project_id": job.id, "page_count": len(project.get("paginas") or []), "workspace": str(root)}


def load_project(root: Path) -> dict[str, Any]:
    path = root / "project.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="project.json nao encontrado")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="project.json invalido") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=422, detail="project.json invalido")
    data.setdefault("paginas", [])
    return data


def save_project(root: Path, project: dict[str, Any]) -> None:
    (root / "project.json").write_text(json.dumps(project, ensure_ascii=True, indent=2), encoding="utf-8")
    state = load_state(root)
    state["dirty"] = True
    save_state(root, state)


def load_state(root: Path) -> dict[str, Any]:
    path = root / "workspace_state.json"
    if not path.exists():
        return {"dirty": False, "preview": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"dirty": False, "preview": {}}
    return data if isinstance(data, dict) else {"dirty": False, "preview": {}}


def save_state(root: Path, state: dict[str, Any]) -> None:
    (root / "workspace_state.json").write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def safe_path(root: Path, asset_path: str) -> Path:
    parts = [part for part in PurePosixPath(asset_path.replace("\\", "/")).parts if part not in {"", ".", "..", "/"}]
    target = (root / Path(*parts)).resolve()
    if root.resolve() != target and root.resolve() not in target.parents:
        raise HTTPException(status_code=400, detail="caminho invalido")
    return target


def relative_asset(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def page_at(project: dict[str, Any], page_index: int) -> dict[str, Any]:
    pages = project.get("paginas") or []
    if page_index < 0 or page_index >= len(pages):
        raise HTTPException(status_code=404, detail="pagina nao encontrada")
    page = pages[page_index]
    if not isinstance(page, dict):
        raise HTTPException(status_code=422, detail="pagina invalida")
    return page


def write_png_layer(root: Path, layer: str, page_index: int, png_data: str | None = None) -> str:
    path = root / "layers" / layer / f"{page_index + 1:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = TRANSPARENT_PNG
    if png_data:
        raw = png_data.split(",", 1)[1] if "," in png_data and "base64" in png_data[:64] else png_data
        try:
            payload = base64.b64decode(raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="png invalido") from exc
    path.write_bytes(payload)
    return relative_asset(root, path)


def _extract_bundle(artifact: Artifact, root: Path, settings: Settings) -> None:
    tmp = root / ".bundle-source.zip"
    get_file(artifact.storage_key, tmp, settings)
    try:
        with zipfile.ZipFile(tmp) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                target = safe_path(root, item.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source:
                    target.write_bytes(source.read())
    finally:
        tmp.unlink(missing_ok=True)


def _copy_artifacts(artifacts: list[Artifact], root: Path, settings: Settings) -> None:
    mapping = {
        "project_json": "project.json",
        "translated_image": "translated",
        "original_image": "originals",
        "input_original": "originals",
        "inpaint_image": "images",
        "layer_mask": "layers/mask",
        "layer_brush": "layers/brush",
        "layer_recovery": "layers/recovery",
        "preview_image": "render-cache/preview",
    }
    for artifact in artifacts:
        dest_rel = mapping.get(artifact.kind)
        if dest_rel is None:
            continue
        dest = root / dest_rel if dest_rel == "project.json" else root / dest_rel / Path(artifact.filename).name
        get_file(artifact.storage_key, dest, settings)


def _extract_input_archive(artifact: Artifact, root: Path, settings: Settings) -> None:
    tmp = root / ".input-source.zip"
    get_file(artifact.storage_key, tmp, settings)
    try:
        with zipfile.ZipFile(tmp) as archive:
            page_index = 0
            for item in sorted(archive.infolist(), key=lambda info: info.filename):
                if item.is_dir() or Path(item.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                page_index += 1
                suffix = Path(item.filename).suffix.lower()
                filename = f"{page_index:03d}{suffix}"
                with archive.open(item) as source:
                    payload = source.read()
                for folder in ["originals", "translated"]:
                    target = safe_path(root, f"{folder}/{filename}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
    finally:
        tmp.unlink(missing_ok=True)


def _build_minimal_project(job: Job, root: Path) -> dict[str, Any]:
    images = sorted((root / "translated").glob("*")) or sorted((root / "originals").glob("*"))
    pages = []
    for index, image in enumerate(path for path in images if path.suffix.lower() in IMAGE_SUFFIXES):
        rel = relative_asset(root, image)
        original = root / "originals" / image.name
        pages.append(
            {
                "index": index,
                "original_path": relative_asset(root, original) if original.exists() else rel,
                "translated_path": rel,
                "rendered_path": rel,
                "text_layers": [],
            }
        )
    if not pages:
        pages.append({"index": 0, "original_path": "", "translated_path": "", "rendered_path": "", "text_layers": []})
    return {
        "job_id": job.id,
        "obra": job.obra,
        "capitulo": job.capitulo,
        "idioma_origem": job.src_lang,
        "idioma_destino": job.dst_lang,
        "config": json.loads(job.config_json) if job.config_json else {},
        "paginas": pages,
    }
