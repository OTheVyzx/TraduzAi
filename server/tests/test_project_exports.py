import json
import struct
import zipfile

from PIL import Image

from server.projects.workspace import project_root
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
    psd_path = project_root(project_id, settings) / "exports" / "page-001.psd"
    psd_bytes = psd_path.read_bytes()
    assert psd_bytes[:4] == b"8BPS"
    assert len(psd_bytes) > 64

    import_zip = tmp_path / "import.zip"
    with zipfile.ZipFile(import_zip, "w") as archive:
        archive.writestr("project.json", json.dumps({"obra": "Importada", "capitulo": "2", "paginas": []}))
    with import_zip.open("rb") as handle:
        imported = client.post("/api/projects/import", files={"file": ("project.zip", handle, "application/zip")})
    assert imported.status_code == 200
    assert imported.json()["project_id"]


def test_psd_page_export_writes_real_layer_section(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200
    root = project_root(project_id, settings)
    Image.new("RGBA", (4, 3), (255, 255, 255, 255)).save(root / "originals" / "001.png")
    Image.new("RGBA", (4, 3), (230, 230, 230, 255)).save(root / "images" / "001-inpaint.png")
    project = json.loads((root / "project.json").read_text(encoding="utf-8"))
    project["paginas"][0]["image_layers"] = {"inpaint": {"path": "images/001-inpaint.png"}}
    (root / "project.json").write_text(json.dumps(project), encoding="utf-8")

    response = client.post(f"/api/projects/{project_id}/exports/psd-page", json={"page_index": 0})
    assert response.status_code == 200
    output = root / "exports" / "page-001.psd"
    payload = output.read_bytes()

    assert payload[:4] == b"8BPS"
    assert struct.unpack(">H", payload[4:6])[0] == 1
    assert struct.unpack(">H", payload[12:14])[0] == 4
    layer_mask_offset = 26 + 4 + 4
    layer_mask_length = struct.unpack(">I", payload[layer_mask_offset : layer_mask_offset + 4])[0]
    assert layer_mask_length > 8
    layer_info_offset = layer_mask_offset + 4
    layer_info_length = struct.unpack(">I", payload[layer_info_offset : layer_info_offset + 4])[0]
    assert layer_info_length > 2
    layer_count = struct.unpack(">h", payload[layer_info_offset + 4 : layer_info_offset + 6])[0]
    assert abs(layer_count) >= 3


def test_zip_full_matches_desktop_export_shape(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200
    root = project_root(project_id, settings)
    (root / "workspace_state.json").write_text("{}", encoding="utf-8")
    (root / "exports" / "old.zip").write_bytes(b"old export")

    response = client.post(f"/api/projects/{project_id}/exports/zip-full")
    assert response.status_code == 200
    output = root / "exports" / f"traduzai-{project_id}.zip"
    with zipfile.ZipFile(output) as archive:
        entries = archive.namelist()
        names = set(entries)
        assert len(entries) == len(names)
        assert "project.json" in names
        assert "translated/001.png" in names
        assert "originals/001.png" in names
        assert "export_manifest.json" in names
        assert "qa_report.json" in names
        assert "workspace_state.json" not in names
        assert "exports/old.zip" not in names
        manifest = json.loads(archive.read("export_manifest.json"))
    assert {entry["path"] for entry in manifest["files"]} >= {"project.json", "translated/001.png", "originals/001.png"}


def test_cbz_uses_app_rendered_image_fields(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200
    root = project_root(project_id, settings)
    for path in (root / "translated").glob("*"):
        path.unlink()
    rendered = root / "render-cache" / "preview" / "001-final.webp"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_bytes(b"final")
    project = json.loads((root / "project.json").read_text(encoding="utf-8"))
    project["paginas"][0].pop("rendered_path", None)
    project["paginas"][0].pop("translated_path", None)
    project["paginas"][0]["arquivo_traduzido"] = "render-cache/preview/001-final.webp"
    (root / "project.json").write_text(json.dumps(project), encoding="utf-8")

    response = client.post(f"/api/projects/{project_id}/exports/cbz")
    assert response.status_code == 200
    output = root / "exports" / f"traduzai-{project_id}.cbz"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["001-final.webp"]
        assert archive.read("001-final.webp") == b"final"

    response = client.post(f"/api/projects/{project_id}/exports/jpg-zip")
    assert response.status_code == 200
    output = root / "exports" / f"traduzai-{project_id}-jpg.zip"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["001-final.webp"]
        assert archive.read("001-final.webp") == b"final"
