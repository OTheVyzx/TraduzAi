from __future__ import annotations

SCHEMA_VERSION = 1


def with_schema_header(payload: dict, *, run_id: str, stage: str, created_at: str | None = None) -> dict:
    if "schema_version" in payload:
        return payload
    header = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "stage": stage}
    if created_at:
        header["created_at"] = created_at
    return {**header, **payload}
