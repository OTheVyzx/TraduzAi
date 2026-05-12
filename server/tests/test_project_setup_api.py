from server.tests.project_test_utils import logged_client, make_settings


def test_setup_context_and_glossary_roundtrip(tmp_path):
    settings = make_settings(tmp_path)
    client = logged_client(settings)

    assert client.get("/api/setup/languages").status_code == 200
    presets = client.get("/api/setup/presets")
    assert presets.status_code == 200
    assert presets.json()["presets"]

    search = client.post("/api/setup/work-search", json={"query": "Probe Real"})
    assert search.status_code == 200
    work = search.json()["results"][0]
    context = client.post("/api/setup/work-context", json={"work_id": work["work_id"], "title": work["title"]})
    assert context.status_code == 200
    assert context.json()["context"]["internet_context"]["glossary_candidates"]

    entry = {"source": "Probe", "target": "Probe", "kind": "nome", "confidence": 1, "status": "reviewed"}
    upsert = client.post(f"/api/setup/glossary/{work['work_id']}/entries", json=entry)
    assert upsert.status_code == 200
    glossary = client.get(f"/api/setup/glossary/{work['work_id']}")
    assert glossary.json()["entries"][0]["source"] == "Probe"

    removed = client.delete(f"/api/setup/glossary/{work['work_id']}/entries/probe")
    assert removed.status_code == 200
    assert client.get(f"/api/setup/glossary/{work['work_id']}").json()["entries"] == []
