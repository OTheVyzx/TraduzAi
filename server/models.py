from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base


Base = declarative_base()


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=new_id)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=new_id)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    owner_user_id = Column(String, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("organization_id", "user_id", name="uq_membership_org_user"),)

    id = Column(String, primary_key=True, default=new_id)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String, nullable=False, default="member")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class WorkerNode(Base):
    __tablename__ = "worker_nodes"

    id = Column(String, primary_key=True, default=new_id)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False, default="offline")
    capabilities_json = Column(Text)
    max_concurrent_jobs = Column(Integer, nullable=False, default=1)
    token_hash = Column(String)
    last_seen_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    id = Column(String, primary_key=True, default=new_id)
    worker_id = Column(String, ForeignKey("worker_nodes.id"), nullable=False, index=True)
    status = Column(String, nullable=False)
    payload_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=new_id)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    worker_id = Column(String, ForeignKey("worker_nodes.id"), index=True)
    status = Column(String, nullable=False, index=True)
    obra = Column(String, nullable=False)
    capitulo = Column(String, nullable=False)
    src_lang = Column(String, nullable=False, default="en")
    dst_lang = Column(String, nullable=False, default="pt-BR")
    mode = Column(String, nullable=False, default="mock")
    config_json = Column(Text)
    page_count = Column(Integer)
    processing_seconds = Column(Float)
    error_code = Column(String)
    error_message = Column(Text)
    cancel_requested_at = Column(DateTime(timezone=True))
    claimed_at = Column(DateTime(timezone=True))
    claimed_until = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    last_heartbeat_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    deleted_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class JobEvent(Base):
    __tablename__ = "job_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    worker_id = Column(String, ForeignKey("worker_nodes.id"))
    stage = Column(String, nullable=False, default="system")
    kind = Column(String, nullable=False, default="status")
    message = Column(Text, nullable=False)
    payload_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, default=new_id)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    kind = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    mime_type = Column(String)
    size = Column(Integer, nullable=False, default=0)
    sha256 = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(String, primary_key=True, default=new_id)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False, index=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)
    event_type = Column(String, nullable=False)
    pages = Column(Integer, nullable=False, default=0)
    processing_seconds = Column(Float)
    estimated_credits = Column(Integer, nullable=False, default=0)
    metadata_json = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=new_id)
    organization_id = Column(String, ForeignKey("organizations.id"), index=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    worker_id = Column(String, ForeignKey("worker_nodes.id"), index=True)
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=False)
    payload_json = Column(Text)
    ip = Column(String)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
