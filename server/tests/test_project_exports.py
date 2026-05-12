import json
import zipfile

from server.tests.project_test_utils import create_completed_project_job, logged_client, make_settings


def test_project_exports_and_import(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200

    full = client.post(f"/api/projects/{project_id}/exports/zip-full")
    assert full.status_code == 200
    cbz = client.post(f"/api/projects/{project_id}/exports/cbz")
    assert cbz.status_code == 200
    jpg = client.post(f"/api/projects/{project_id}/exports/jpg-zip")
    assert jpg.status_code == 200
    psd = client.post(f"/api/projects/{project_id}/exports/psd-page", json={"page_index": 0})
    assert psd.status_code == 200

    import_zip = tmp_path / "import.zip"
    with zipfile.ZipFile(import_zip, "w") as archive:
        archive.writestr("project.json", json.dumps({"obra": "Importada", "capitulo": "2", "paginas": []}))
    with import_zip.open("rb") as handle:
        imported = client.post("/api/projects/import", files={"file": ("project.zip", handle, "application/zip")})
    assert imported.status_code == 200
    assert imported.json()["project_id"]
