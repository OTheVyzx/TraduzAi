"""The only authority allowed to invoke band-first compatibility behavior."""

from __future__ import annotations

from typing import Any, Callable


LEGACY_UNVERIFIED = "legacy_unverified"
VERIFIED = "verified"


def owner_mode_for_project(project: dict[str, Any]) -> str:
    status = str(project.get("owner_graph_status") or "").strip().lower()
    if status == VERIFIED:
        return "enforce"
    if status == LEGACY_UNVERIFIED:
        return "legacy"
    raise ValueError("project must declare verified or legacy_unverified ownership")


class LegacyUnverifiedAdapter:
    """Quarantine legacy semantics without granting owner-graph trust."""

    def __init__(self, project: dict[str, Any]) -> None:
        if owner_mode_for_project(project) != "legacy":
            raise ValueError("legacy adapter requires a legacy_unverified project")
        self.project = project

    @property
    def can_use_band_semantics(self) -> bool:
        return True

    def call(self, helper: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if str(self.project.get("owner_graph_status") or "") != LEGACY_UNVERIFIED:
            raise ValueError("legacy adapter lost its legacy_unverified boundary")
        return helper(*args, **kwargs)

    def mark_verified(self) -> None:
        raise ValueError("legacy adapter cannot mark an owner graph verified")
