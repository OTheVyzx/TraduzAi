from server.tests.project_test_utils import create_completed_project_job, logged_client, make_settings


def test_materialize_preview_and_asset_serving(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)
    project_id = create_completed_project_job(settings, tmp_path)

    materialize = client.post(f"/api/jobs/{project_id}/materialize-project")
    assert materialize.status_code == 200
    assert materialize.json()["page_count"] == 1

    project = client.get(f"/api/projects/{project_id}")
    assert project.status_code == 200
    assert project.json()["project"]["paginas"][0]["text_layers"][0]["id"] == "layer-1"

    page = client.get(f"/api/projects/{project_id}/pages/0")
    assert page.status_code == 200
    assert "rendered" in page.json()["layers"]

    preview = client.post(f"/api/projects/{project_id}/pages/0/render-preview")
    assert preview.status_code == 200
    asset = client.get(f"/api/projects/{project_id}/assets/{preview.json()['asset_path']}")
    assert asset.status_code == 200

    traversal = client.get(f"/api/projects/{project_id}/assets/../../secret.txt")
    assert traversal.status_code in {400, 404}
