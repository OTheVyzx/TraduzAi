from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings
from server.db import session_scope
from server.models import Artifact, Job, new_id
from server.orgs import default_org_for_user
from server.storage import put_file


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
    )


def logged_client(settings: Settings) -> TestClient:
    client = TestClient(create_app(settings))
    login = client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"})
    assert login.status_code == 200
    return client


def create_completed_project_job(settings: Settings, tmp_path: Path) -> str:
    project_id = new_id()
    project = {
        "obra": "Obra",
        "capitulo": "1",
        "idioma_origem": "en",
        "idioma_destino": "pt-BR",
        "paginas": [
            {
                "index": 0,
                "original_path": "originals/001.png",
                "translated_path": "translated/001.png",
                "rendered_path": "translated/001.png",
                "text_layers": [{"id": "layer-1", "texto": "Oi", "x": 10, "y": 20, "w": 80, "h": 40}],
            }
        ],
    }
    project_file = tmp_path / "project.json"
    translated = tmp_path / "001.png"
    original = tmp_path / "orig.png"
    project_file.write_text(json.dumps(project), encoding="utf-8")
    translated.write_bytes(b"\x89PNG\r\n\x1a\ntranslated")
    original.write_bytes(b"\x89PNG\r\n\x1a\noriginal")
    with session_scope(settings) as db:
        user = db.query(__import__("server.models", fromlist=["User"]).User).filter_by(email="admin@local").one()
        org = default_org_for_user(db, user.id)
        assert org is not None
        job = Job(
            id=project_id,
            organization_id=org.id,
            user_id=user.id,
            status="completed",
            obra="Obra",
            capitulo="1",
            src_lang="en",
            dst_lang="pt-BR",
            mode="real",
            page_count=1,
        )
        db.add(job)
        db.flush()
        for kind, path, filename, storage_key in [
            ("project_json", project_file, "project.json", f"jobs/{project_id}/project.json"),
            ("translated_image", translated, "001.png", f"jobs/{project_id}/translated/001.png"),
            ("original_image", original, "001.png", f"jobs/{project_id}/originals/001.png"),
        ]:
            put_file(path, storage_key, "application/octet-stream", settings)
            db.add(
                Artifact(
                    job_id=project_id,
                    organization_id=org.id,
                    kind=kind,
                    storage_key=storage_key,
                    filename=filename,
                    mime_type="application/octet-stream",
                    size=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return project_id
