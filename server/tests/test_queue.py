from server.config import Settings
from server.db import bootstrap_database, session_scope
from server.models import Job, Organization, User, WorkerNode
from server.queue import claim_next, claim_specific, enqueue


def make_settings(tmp_path):
    return Settings(
        env="dev",
        database_url=f"sqlite:///{tmp_path / 'app.db'}",
        storage_dir=tmp_path / "storage",
        admin_email="admin@local",
        admin_password="secret123",
        worker_token="dev-token",
    )


def add_job(session, job_id, org_id, user_id, status="queued", worker_id=None):
    session.add(
        Job(
            id=job_id,
            organization_id=org_id,
            user_id=user_id,
            worker_id=worker_id,
            status=status,
            obra="Obra",
            capitulo="1",
            src_lang="en",
            dst_lang="pt-BR",
            mode="mock",
        )
    )


def test_claim_next_respects_worker_concurrency(tmp_path):
    settings = make_settings(tmp_path)
    bootstrap_database(settings)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        session.add(WorkerNode(id="worker-1", name="admin-pc", status="online", max_concurrent_jobs=1))
        add_job(session, "job-1", org.id, admin.id)
        add_job(session, "job-2", org.id, admin.id)

    claimed = claim_next(settings, "worker-1", {})
    assert claimed is not None
    assert claimed.id == "job-1"

    blocked = claim_next(settings, "worker-1", {})
    assert blocked is None

    with session_scope(settings) as session:
        job_1 = session.get(Job, "job-1")
        job_2 = session.get(Job, "job-2")
        assert job_1.status == "claimed"
        assert job_1.worker_id == "worker-1"
        assert job_1.claimed_until is not None
        assert job_2.status == "queued"


def test_enqueue_is_idempotent(tmp_path):
    settings = make_settings(tmp_path)
    bootstrap_database(settings)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        add_job(session, "job-1", org.id, admin.id, status="failed")

    enqueue(settings, "job-1")
    enqueue(settings, "job-1")

    with session_scope(settings) as session:
        assert session.get(Job, "job-1").status == "queued"


def test_claim_next_respects_worker_mode_capabilities(tmp_path):
    settings = make_settings(tmp_path)
    bootstrap_database(settings)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        session.add(WorkerNode(id="worker-1", name="admin-pc", status="online", max_concurrent_jobs=1))
        add_job(session, "job-real", org.id, admin.id)
        session.flush()
        session.get(Job, "job-real").mode = "real"

    assert claim_next(settings, "worker-1", {"mode": ["mock"]}) is None
    claimed = claim_next(settings, "worker-1", {"mode": ["real"]})
    assert claimed is not None
    assert claimed.id == "job-real"


def test_claim_specific_only_claims_requested_job(tmp_path):
    settings = make_settings(tmp_path)
    bootstrap_database(settings)

    with session_scope(settings) as session:
        admin = session.query(User).filter_by(email="admin@local").one()
        org = session.query(Organization).filter_by(slug="default").one()
        session.add(WorkerNode(id="worker-1", name="serverless", status="online", max_concurrent_jobs=1))
        add_job(session, "job-old", org.id, admin.id)
        add_job(session, "job-target", org.id, admin.id)

    claimed = claim_specific(settings, "worker-1", "job-target", {"mode": ["mock"]})

    assert claimed is not None
    assert claimed.id == "job-target"
    with session_scope(settings) as session:
        assert session.get(Job, "job-old").status == "queued"
        assert session.get(Job, "job-target").status == "claimed"
