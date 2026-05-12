from server.auth import verify_password
from server.config import Settings
from server.db import bootstrap_database, session_scope
from server.models import Membership, Organization, Setting, User


def test_bootstrap_creates_admin_default_org_membership_and_session_secret(tmp_path):
    settings = Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
    )

    bootstrap_database(settings)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        assert admin.role == "admin"
        assert admin.is_active is True
        assert verify_password("secret123", admin.password_hash)

        org = session.query(Organization).filter_by(slug="default").one()
        assert org.owner_user_id == admin.id

        membership = session.query(Membership).filter_by(organization_id=org.id, user_id=admin.id).one()
        assert membership.role == "owner"

        secret = session.query(Setting).filter_by(key="session_secret").one()
        assert len(secret.value) >= 32


def test_bootstrap_marks_active_jobs_failed_after_restart(tmp_path):
    settings = Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
    )
    bootstrap_database(settings)

    from server.models import Job

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        session.add(
            Job(
                id="job-running",
                organization_id=org.id,
                user_id=admin.id,
                status="running",
                obra="Obra",
                capitulo="1",
                src_lang="en",
                dst_lang="pt-BR",
                mode="mock",
            )
        )

    bootstrap_database(settings)

    with session_scope(settings) as session:
        job = session.get(Job, "job-running")
        assert job.status == "failed"
        assert job.error_code == "restart"
