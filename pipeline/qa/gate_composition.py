"""Pure, normalized composition of functional and style export gates."""

from __future__ import annotations

import copy
from typing import Any


def qa_integrity_issue(detail: Any, *, scope: str) -> dict[str, Any]:
    """Create the canonical fail-closed issue for an unavailable QA subgate."""

    return {
        "code": "qa_integrity_failure",
        "reason": "qa_integrity_failure",
        "type": "qa_integrity_failure",
        "issue_scope": scope,
        "severity": "critical",
        "blocks_export": True,
        "flags": ["qa_integrity_failure"],
        "owner_ids": [],
        "offenders": [str(detail)],
    }


def _issues(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


def _blocking(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        issue
        for issue in issues
        if str(issue.get("severity") or "").lower() == "critical"
        or bool(issue.get("blocks_export"))
    ]


def normalize_export_gate(
    issues: list[dict[str, Any]],
    *,
    blocked: bool,
    review: bool = False,
    override: bool = False,
    functional_gate: dict[str, Any] | None = None,
    style_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute every derived gate field from canonical issues and subgate state."""

    canonical = _issues(issues)
    critical = [item for item in canonical if str(item.get("severity") or "").lower() == "critical"]
    warnings = [item for item in canonical if str(item.get("severity") or "").lower() == "warning"]
    blocking = _blocking(canonical)
    effective_blocked = bool(blocked or blocking)
    status = "OVERRIDDEN" if effective_blocked and override else "BLOCK" if effective_blocked else "REVIEW" if review else "PASS"
    result = {
        "status": status,
        "allowed": status != "BLOCK",
        "override": bool(override and effective_blocked),
        "issue_count": len(canonical),
        "blocking_issue_count": len(blocking),
        "blocking_flag_count": sum(len(item.get("flags") or []) for item in blocking),
        "critical_issue_count": len(critical),
        "critical_flag_count": sum(len(item.get("flags") or []) for item in critical),
        "review_issue_count": len(warnings),
        "review_flag_count": sum(len(item.get("flags") or []) for item in warnings),
        "needs_review": bool(warnings or effective_blocked),
        "issues": canonical,
    }
    if functional_gate is not None or style_gate is not None:
        functional = copy.deepcopy(functional_gate or {"status": "PASS"})
        style = copy.deepcopy(style_gate or {"status": "PASS"})
        result["subgates"] = {
            "functional": {
                "status": str(functional.get("status") or "PASS").upper(),
                "allowed": functional.get("allowed") is not False,
            },
            "style": {
                "status": str(style.get("status") or "PASS").upper(),
                "would_block": bool(style.get("would_block")),
                "blocking_owner_ids": sorted(
                    {str(value) for value in style.get("blocking_owner_ids") or []}
                ),
            },
        }
    return result


def compose_export_gate(
    functional_gate: dict[str, Any],
    style_gate: dict[str, Any],
    *,
    override: bool = False,
) -> dict[str, Any]:
    """Return the conjunction of normalized functional and style subgates."""

    functional = copy.deepcopy(functional_gate or {})
    style = copy.deepcopy(style_gate or {})
    functional_status = str(functional.get("status") or "BLOCK").upper()
    style_status = str(style.get("status") or "BLOCK").upper()
    functional_blocked = functional_status == "BLOCK" or functional.get("allowed") is False
    style_blocked = style_status == "BLOCK"
    issues = _issues(functional.get("issues")) + _issues(style.get("issues"))
    if style_blocked and not _issues(style.get("issues")):
        issues.append(
            {
                "code": "style_fidelity_high_confidence_mismatch",
                "reason": "style_fidelity_high_confidence_mismatch",
                "type": "style_fidelity",
                "issue_scope": "owner",
                "severity": "critical",
                "blocks_export": True,
                "flags": ["style_fidelity_high_confidence_mismatch"],
                "owner_ids": sorted(
                    {str(value) for value in style.get("blocking_owner_ids") or []}
                ),
            }
        )
    return normalize_export_gate(
        issues,
        blocked=functional_blocked or style_blocked,
        review=functional_status == "REVIEW",
        override=override,
        functional_gate=functional,
        style_gate=style,
    )
