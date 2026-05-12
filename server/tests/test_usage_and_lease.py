from datetime import datetime, timedelta, timezone

from server.config import Settings
from server.db import bootstrap_database, session_scope
from server.models import Job, Organization, UsageEvent, User, WorkerNode
from server.workers.lease import fail_lost_jobs


def test_fail_lost_jobs_marks_worker_lost_and_records_usage(tmp_path):
    settings = Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
        lease_timeout_seconds=60,
    )
    bootstrap_database(settings)
    stale = datetime.now(timezone.utc) - timedelta(minutes=5)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        session.add(WorkerNode(id="worker-1", name="admin-pc", status="online", max_concurrent_jobs=1))
        session.flush()
        session.add(
            Job(
                id="job-lost",
                organization_id=org.id,
                user_id=admin.id,
                worker_id="worker-1",
                status="running",
                obra="Obra",
                capitulo="1",
                src_lang="en",
                dst_lang="pt-BR",
                mode="mock",
                page_count=3,
                last_heartbeat_at=stale,
            )
        )

    changed = fail_lost_jobs(settings)
    assert changed == 1

    with session_scope(settings) as session:
        job = session.get(Job, "job-lost")
        assert job.status == "failed"
        assert job.error_code == "worker_lost"
        usage = session.query(UsageEvent).filter_by(job_id="job-lost").one()
        assert usage.event_type == "job_failed"
        assert usage.pages == 3
