from __future__ import annotations

import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from server.auth import hash_password
from server.config import Settings
from server.models import Base, Job, Membership, Organization, Setting, User, utc_now


_engines = {}
_makers = {}


def get_engine(settings: Settings):
    engine = _engines.get(settings.database_url)
    if engine is not None:
        return engine
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url, future=True, connect_args={"check_same_thread": False})
    if settings.database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    _engines[settings.database_url] = engine
    _makers[settings.database_url] = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    return engine


def get_sessionmaker(settings: Settings):
    get_engine(settings)
    return _makers[settings.database_url]


@contextmanager
def session_scope(settings: Settings) -> Iterator[OrmSession]:
    maker = get_sessionmaker(settings)
    session = maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def bootstrap_database(settings: Settings) -> None:
    if settings.env != "dev" and not settings.worker_token:
        raise RuntimeError("TRADUZAI_WORKER_TOKEN obrigatorio fora de dev")
    if not settings.admin_password:
        if settings.env == "dev":
            settings.admin_password = "admin"
        else:
            raise RuntimeError("TRADUZAI_ADMIN_PASSWORD obrigatorio no primeiro boot")

    engine = get_engine(settings)
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        if settings.database_url.startswith("sqlite"):
            columns = {row[1] for row in conn.execute(text("PRAGMA table_info(jobs)")).fetchall()}
            if "config_json" not in columns:
                conn.execute(text("ALTER TABLE jobs ADD COLUMN config_json TEXT"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_org_status_created ON jobs (organization_id, status, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_user_created ON jobs (user_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_status_created ON jobs (status, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_jobs_worker_status ON jobs (worker_id, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_job_events_job_created ON job_events (job_id, created_at)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_artifacts_job ON artifacts (job_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_memberships_user ON memberships (user_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_usage_org_created ON usage_events (organization_id, created_at)"))

    with session_scope(settings) as session:
        _ensure_setting(session, "schema_version", "1")
        _ensure_setting(session, "session_secret", secrets.token_urlsafe(48))
        admin = session.query(User).filter_by(email=settings.admin_email.lower()).one_or_none()
        if admin is None:
            admin = User(
                email=settings.admin_email.lower(),
                password_hash=hash_password(settings.admin_password),
                role="admin",
                is_active=True,
            )
            session.add(admin)
            session.flush()
        org = session.query(Organization).filter_by(slug="default").one_or_none()
        if org is None:
            org = Organization(name="Default", slug="default", owner_user_id=admin.id)
            session.add(org)
            session.flush()
        membership = session.query(Membership).filter_by(organization_id=org.id, user_id=admin.id).one_or_none()
        if membership is None:
            session.add(Membership(organization_id=org.id, user_id=admin.id, role="owner"))

        active = session.query(Job).filter(Job.status.in_(["claimed", "running", "uploading_results"])).all()
        for job in active:
            job.status = "failed"
            job.error_code = "restart"
            job.error_message = "Servidor reiniciado durante o processamento"
            job.finished_at = utc_now()


def _ensure_setting(session: OrmSession, key: str, value: str) -> None:
    if session.get(Setting, key) is None:
        session.add(Setting(key=key, value=value))
