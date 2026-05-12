from __future__ import annotations

import secrets
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from server.models import Session, User, utc_now


_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_session(db, user: User) -> Session:
    session = Session(id=secrets.token_urlsafe(32), user_id=user.id, expires_at=utc_now() + timedelta(days=14))
    db.add(session)
    return session
