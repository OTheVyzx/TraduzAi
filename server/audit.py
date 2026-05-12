from __future__ import annotations

import json

from server.models import AuditLog


def log_event(db, *, action: str, entity_type: str, entity_id: str, organization_id=None, user_id=None, worker_id=None, payload=None, ip=None):
    db.add(
        AuditLog(
            organization_id=organization_id,
            user_id=user_id,
            worker_id=worker_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload_json=json.dumps(payload or {}, ensure_ascii=True),
            ip=ip,
        )
    )
