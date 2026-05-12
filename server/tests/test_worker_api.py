import io
import json
import zipfile

from fastapi.testclient import TestClient

from server.app import create_app
from server.config import Settings


def make_settings(tmp_path):
    return Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
    )


def test_worker_claim_exposes_protected_input_download(tmp_path):
    settings = make_settings(tmp_path)
    client = TestClient(create_app(settings))
    login = client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"})
    assert login.status_code == 200

    image = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    upload = client.post(
        "/api/jobs",
        data={"obra": "Obra", "capitulo": "1", "mode": "real"},
        files={"file": ("page.png", image, "image/png")},
    )
    assert upload.status_code == 200
    job_id = upload.json()["job"]["id"]

    headers = {"Authorization": "Bearer dev-token"}
    register = client.post("/api/workers/register", headers=headers, json={"name": "admin-pc"})
    assert register.status_code == 200
    worker_id = register.json()["worker_id"]

    claim = client.post("/api/workers/claim-job", headers=headers, json={"worker_id": worker_id, "capabilities": {}})
    assert claim.status_code == 200
    job = claim.json()["job"]
    assert job["id"] == job_id
    assert job["input_download_url"] == f"/api/workers/jobs/{job_id}/input"
    assert job["input_filename"] == "page.png"

    download = client.get(job["input_download_url"], headers=headers, params={"worker_id": worker_id})
    assert download.status_code == 200
    assert download.content == image


def test_manual_project_finishes_without_worker_and_materializes(tmp_path):
    settings = make_settings(tmp_path)
    client = TestClient(create_app(settings))
    login = client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"})
    assert login.status_code == 200

    image = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    upload = client.post(
        "/api/jobs",
        data={
            "obra": "Obra",
            "capitulo": "1",
            "mode": "real",
            "project_config": json.dumps({"mode": "manual"}),
        },
        files={"file": ("page.png", image, "image/png")},
    )
    assert upload.status_code == 200
    job = upload.json()["job"]
    assert job["status"] == "completed"

    headers = {"Authorization": "Bearer dev-token"}
    worker_id = client.post("/api/workers/register", headers=headers, json={"name": "admin-pc"}).json()["worker_id"]
    claim = client.post("/api/workers/claim-job", headers=headers, json={"worker_id": worker_id, "capabilities": {"mode": ["real"]}})
    assert claim.status_code == 200
    assert claim.json()["job"] is None

    materialized = client.post(f"/api/jobs/{job['id']}/materialize-project")
    assert materialized.status_code == 200
    assert materialized.json()["project_id"] == job["id"]
    project = client.get(f"/api/projects/{job['id']}")
    assert project.status_code == 200
    assert project.json()["project"]["paginas"][0]["original_path"] == "originals/page.png"


def test_manual_archive_materializes_pages(tmp_path):
    settings = make_settings(tmp_path)
    client = TestClient(create_app(settings))
    login = client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"})
    assert login.status_code == 200

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("b-page.png", b"\x89PNG\r\n\x1a\n" + b"b" * 8)
        archive.writestr("a-page.png", b"\x89PNG\r\n\x1a\n" + b"a" * 8)
    upload = client.post(
        "/api/jobs",
        data={"mode": "real", "project_config": json.dumps({"mode": "manual"})},
        files={"file": ("chapter.cbz", archive_bytes.getvalue(), "application/zip")},
    )
    assert upload.status_code == 200
    job_id = upload.json()["job"]["id"]

    materialized = client.post(f"/api/jobs/{job_id}/materialize-project")
    assert materialized.status_code == 200
    assert materialized.json()["page_count"] == 2
    project = client.get(f"/api/projects/{job_id}")
    assert project.status_code == 200
    pages = project.json()["project"]["paginas"]
    assert [page["original_path"] for page in pages] == ["originals/001.png", "originals/002.png"]
    assert [page["translated_path"] for page in pages] == ["translated/001.png", "translated/002.png"]


def test_failed_job_can_be_retried_with_same_input(tmp_path):
    settings = make_settings(tmp_path)
    client = TestClient(create_app(settings))
    login = client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"})
    assert login.status_code == 200

    image = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    upload = client.post(
        "/api/jobs",
        data={"obra": "Obra", "capitulo": "1", "mode": "real"},
        files={"file": ("page.png", image, "image/png")},
    )
    assert upload.status_code == 200
    job_id = upload.json()["job"]["id"]

    headers = {"Authorization": "Bearer dev-token"}
    worker_id = client.post("/api/workers/register", headers=headers, json={"name": "admin-pc"}).json()["worker_id"]
    assert client.post("/api/workers/claim-job", headers=headers, json={"worker_id": worker_id, "capabilities": {"mode": ["real"]}}).status_code == 200
    artifact = client.post(
        f"/api/workers/jobs/{job_id}/artifact",
        headers=headers,
        data={"worker_id": worker_id, "kind": "runner_log"},
        files={"file": ("runner.log", b"falhou", "text/plain")},
    )
    assert artifact.status_code == 200
    failed = client.post(
        f"/api/workers/jobs/{job_id}/fail",
        headers=headers,
        json={"worker_id": worker_id, "error_code": "worker_error", "error_message": "reset"},
    )
    assert failed.status_code == 200

    retry = client.post(f"/api/jobs/{job_id}/retry")
    assert retry.status_code == 200
    retried = retry.json()["job"]
    assert retried["status"] == "queued"
    assert retried["error_code"] is None
    assert retried["error_message"] is None

    detail = client.get(f"/api/jobs/{job_id}")
    assert detail.status_code == 200
    artifacts = detail.json()["job"]["artifacts"]
    assert [item["kind"] for item in artifacts] == ["input_original"]
