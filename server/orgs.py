from __future__ import annotations

from server.models import Membership, Organization


def default_org_for_user(db, user_id: str) -> Organization | None:
    membership = db.query(Membership).filter_by(user_id=user_id).order_by(Membership.created_at.asc()).first()
    if membership is None:
        return None
    return db.get(Organization, membership.organization_id)


def user_belongs_to_org(db, user_id: str, organization_id: str) -> bool:
    return db.query(Membership).filter_by(user_id=user_id, organization_id=organization_id).first() is not None
