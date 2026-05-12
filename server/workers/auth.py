from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException

from server.config import Settings
from server.deps import get_settings


def require_worker_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.worker_token
    if not expected:
        if settings.env == "dev":
            expected = "dev-token"
        else:
            raise HTTPException(status_code=503, detail="worker token ausente")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="worker token obrigatorio")
    supplied = authorization[len(prefix) :]
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="worker token invalido")
