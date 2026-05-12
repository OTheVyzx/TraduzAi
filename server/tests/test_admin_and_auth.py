from fastapi.testclient import TestClient

from server.app import create_app
from server.auth_api import _attempts
from server.config import Settings
from server.db import session_scope
from server.models import User


def make_settings(tmp_path):
    return Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        google_client_id="google-client",
        google_client_secret="google-secret",
        worker_token="dev-token",
    )


def test_login_rate_limit_blocks_after_five_bad_attempts(tmp_path):
    _attempts.clear()
    client = TestClient(create_app(make_settings(tmp_path)))
    for _ in range(5):
        response = client.post("/api/auth/login", json={"email": "admin@local", "password": "wrong"})
        assert response.status_code == 401

    blocked = client.post("/api/auth/login", json={"email": "admin@local", "password": "wrong"})
    assert blocked.status_code == 429


def test_admin_overview_lists_jobs_workers_and_audit(tmp_path):
    _attempts.clear()
    client = TestClient(create_app(make_settings(tmp_path)))
    assert client.post("/api/auth/login", json={"email": "admin@local", "password": "secret123"}).status_code == 200
    headers = {"Authorization": "Bearer dev-token"}
    assert client.post("/api/workers/register", headers=headers, json={"name": "admin-pc"}).status_code == 200
    image = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    assert (
        client.post(
            "/api/jobs",
            data={"obra": "Obra", "capitulo": "1", "mode": "mock"},
            files={"file": ("page.png", image, "image/png")},
        ).status_code
        == 200
    )

    response = client.get("/api/admin/overview")
    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"][0]["obra"] == "Obra"
    assert payload["workers"][0]["name"] == "admin-pc"
    assert any(item["action"] == "job.created" for item in payload["audit_logs"])


def test_google_start_redirects_to_google_and_sets_state(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path)))
    response = client.get("/api/auth/google/start?next=/dashboard", follow_redirects=False)
    assert response.status_code == 302
    assert "accounts.google.com" in response.headers["location"]
    assert "google_oauth_state=" in response.headers.get("set-cookie", "")


def test_google_callback_creates_session_and_user(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    client = TestClient(create_app(settings))

    monkeypatch.setattr(
        "server.auth_api._exchange_google_code",
        lambda code, redirect_uri, settings: {"access_token": "google-token"},
    )
    monkeypatch.setattr(
        "server.auth_api._fetch_google_userinfo",
        lambda access_token: {"email": "google-user@example.com", "email_verified": True},
    )

    start = client.get("/api/auth/google/start?next=/dashboard", follow_redirects=False)
    state_cookie = client.cookies.get("google_oauth_state")
    assert start.status_code == 302
    assert state_cookie

    callback = client.get(
        f"/api/auth/google/callback?code=test-code&state={state_cookie}",
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert callback.headers["location"].endswith("/dashboard")
    assert "session_id" in callback.headers.get("set-cookie", "")

    with session_scope(settings) as db:
        created_user = db.query(User).filter_by(email="google-user@example.com").one_or_none()
        assert created_user is not None
