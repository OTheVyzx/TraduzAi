from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
import json
import secrets
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from server.auth import create_session, hash_password, verify_password
from server.config import Settings
from server.db import session_scope
from server.deps import current_user, get_settings
from server.models import Session, User, utc_now


router = APIRouter(prefix="/api/auth", tags=["auth"])
_attempts: dict[str, deque] = defaultdict(deque)
GOOGLE_STATE_COOKIE = "google_oauth_state"
GOOGLE_NEXT_COOKIE = "google_oauth_next"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


class LoginRequest(BaseModel):
    email: str
    password: str


def _check_rate_limit(ip: str) -> None:
    now = utc_now()
    window = now - timedelta(minutes=15)
    attempts = _attempts[ip]
    while attempts and attempts[0] < window:
        attempts.popleft()
    if len(attempts) >= 5:
        raise HTTPException(status_code=429, detail="muitas tentativas")


def _set_session_cookie(response: Response, settings: Settings, session_id: str) -> None:
    response.set_cookie(
        "session_id",
        session_id,
        httponly=True,
        samesite="strict",
        secure=settings.env == "prod",
        max_age=14 * 24 * 3600,
    )


def _exchange_google_code(code: str, redirect_uri: str, settings: Settings) -> dict:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=503, detail="google login indisponivel")
    payload = urlencode(
        {
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = UrlRequest(
        GOOGLE_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_google_userinfo(access_token: str) -> dict:
    request = UrlRequest(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@router.post("/login")
def login(payload: LoginRequest, request: Request, response: Response, settings: Settings = Depends(get_settings)):
    ip = request.client.host if request.client else "local"
    _check_rate_limit(ip)
    with session_scope(settings) as db:
        user = db.query(User).filter_by(email=payload.email.lower()).one_or_none()
        if user is None or not verify_password(payload.password, user.password_hash):
            _attempts[ip].append(utc_now())
            raise HTTPException(status_code=401, detail="email ou senha invalidos")
        session = create_session(db, user)
        _set_session_cookie(response, settings, session.id)
        return {"user": {"id": user.id, "email": user.email, "role": user.role}}


@router.get("/google/start")
def google_start(request: Request, next: str = "/dashboard", settings: Settings = Depends(get_settings)):
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=503, detail="google login indisponivel")
    state = secrets.token_urlsafe(24)
    redirect_uri = str(request.url_for("google_callback"))
    auth_query = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    response = RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{auth_query}", status_code=302)
    response.set_cookie(
        GOOGLE_STATE_COOKIE,
        state,
        httponly=True,
        samesite="lax",
        secure=settings.env == "prod",
        max_age=10 * 60,
    )
    response.set_cookie(
        GOOGLE_NEXT_COOKIE,
        next if next.startswith("/") else "/dashboard",
        httponly=True,
        samesite="lax",
        secure=settings.env == "prod",
        max_age=10 * 60,
    )
    return response


@router.get("/google/callback", name="google_callback")
def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    settings: Settings = Depends(get_settings),
):
    expected_state = request.cookies.get(GOOGLE_STATE_COOKIE)
    next_path = request.cookies.get(GOOGLE_NEXT_COOKIE, "/dashboard")
    if not code or not state or not expected_state or state != expected_state:
        redirect = RedirectResponse(url=f"{settings.site_origin}/login?error=google_state", status_code=302)
        redirect.delete_cookie(GOOGLE_STATE_COOKIE)
        redirect.delete_cookie(GOOGLE_NEXT_COOKIE)
        return redirect

    token_payload = _exchange_google_code(code, str(request.url_for("google_callback")), settings)
    access_token = token_payload.get("access_token")
    if not access_token:
        redirect = RedirectResponse(url=f"{settings.site_origin}/login?error=google_token", status_code=302)
        redirect.delete_cookie(GOOGLE_STATE_COOKIE)
        redirect.delete_cookie(GOOGLE_NEXT_COOKIE)
        return redirect

    userinfo = _fetch_google_userinfo(access_token)
    email = (userinfo.get("email") or "").strip().lower()
    if not email or userinfo.get("email_verified") is False:
        redirect = RedirectResponse(url=f"{settings.site_origin}/login?error=google_email", status_code=302)
        redirect.delete_cookie(GOOGLE_STATE_COOKIE)
        redirect.delete_cookie(GOOGLE_NEXT_COOKIE)
        return redirect

    with session_scope(settings) as db:
        user = db.query(User).filter_by(email=email).one_or_none()
        if user is None:
            user = User(
                email=email,
                password_hash=hash_password(secrets.token_urlsafe(24)),
                role="user",
                is_active=True,
            )
            db.add(user)
            db.flush()
        session = create_session(db, user)
        redirect = RedirectResponse(url=f"{settings.site_origin}{next_path if next_path.startswith('/') else '/dashboard'}", status_code=302)
        _set_session_cookie(redirect, settings, session.id)
        redirect.delete_cookie(GOOGLE_STATE_COOKIE)
        redirect.delete_cookie(GOOGLE_NEXT_COOKIE)
        return redirect


@router.post("/logout")
def logout(response: Response, session_id: str | None = None, settings: Settings = Depends(get_settings)):
    if session_id:
        with session_scope(settings) as db:
            session = db.get(Session, session_id)
            if session is not None:
                db.delete(session)
    response.delete_cookie("session_id")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"user": {"id": user.id, "email": user.email, "role": user.role}}
