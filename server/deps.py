from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException, Request

from server.config import Settings, load_settings
from server.db import session_scope
from server.models import Session, User, utc_now


def _aware(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value


def get_settings(request: Request) -> Settings:
    if hasattr(request.app.state, "settings"):
        return request.app.state.settings
    return load_settings()


def current_user(session_id: str | None = Cookie(default=None), settings: Settings = Depends(get_settings)) -> User:
    if not session_id:
        raise HTTPException(status_code=401, detail="login obrigatorio")
    with session_scope(settings) as db:
        session = db.get(Session, session_id)
        if session is None or _aware(session.expires_at) < utc_now():
            raise HTTPException(status_code=401, detail="sessao expirada")
        user = db.get(User, session.user_id)
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="usuario invalido")
        db.expunge(user)
        return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin obrigatorio")
    return user
