import base64

from server.projects.workspace import TRANSPARENT_PNG
from server.tests.project_test_utils import create_completed_project_job, logged_client, make_settings


def test_editor_text_and_bitmap_updates(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200

    patch = client.patch(
        f"/api/projects/{project_id}/editor/pages/0/text-layers/layer-1",
        json={"patch": {"texto": "Ola", "x": 30}},
    )
    assert patch.status_code == 200
    assert patch.json()["layer"]["texto"] == "Ola"

    create = client.post(f"/api/projects/{project_id}/editor/pages/0/text-layers", json={"layer": {"texto": "Novo"}})
    assert create.status_code == 200
    new_id = create.json()["layer"]["id"]
    assert client.delete(f"/api/projects/{project_id}/editor/pages/0/text-layers/{new_id}").status_code == 200

    mask = client.post(f"/api/projects/{project_id}/editor/pages/0/mask", json={})
    assert mask.status_code == 200
    assert mask.json()["asset_path"].endswith("layers/mask/001.png")

    action = client.post(f"/api/projects/{project_id}/editor/pages/0/actions", json={"action": "ocr", "region": {"bbox": [0, 0, 20, 20]}})
    assert action.status_code == 200


def test_recovery_without_png_does_not_replace_inpaint_with_transparent_layer(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200

    recovery = client.post(
        f"/api/projects/{project_id}/editor/pages/0/recovery",
        json={"width": 100, "height": 100, "brush_size": 20, "strokes": [[[10, 10], [20, 20]]]},
    )

    assert recovery.status_code == 200
    assert recovery.json()["asset_path"] == "translated/001.png"


def test_recovery_with_png_updates_visible_inpaint_layer(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)
    assert client.post(f"/api/jobs/{project_id}/materialize-project").status_code == 200
    png_data = "data:image/png;base64," + base64.b64encode(TRANSPARENT_PNG).decode("ascii")

    recovery = client.post(
        f"/api/projects/{project_id}/editor/pages/0/recovery",
        json={"png_data": png_data, "width": 1, "height": 1, "brush_size": 20, "strokes": [[[0, 0]]]},
    )

    assert recovery.status_code == 200
    assert recovery.json()["asset_path"].endswith("layers/recovery/001.png")
    page = client.get(f"/api/projects/{project_id}/editor/pages/0").json()["page"]
    assert page["image_layers"]["inpaint"]["path"].endswith("layers/recovery/001.png")
