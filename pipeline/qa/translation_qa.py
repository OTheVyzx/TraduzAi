"""Translation QA flags, severity and blocking policies."""

from __future__ import annotations

from collections import Counter
from typing import Any

FLAG_SEVERITY = {
    "glossary_violation": "critical",
    "forbidden_translation": "critical",
    "entity_mistranslated": "critical",
    "placeholder_lost": "critical",
    "schema_validation_error": "critical",
    "empty_translation": "high",
    "untranslated_english": "high",
    "gibberish_detected": "high",
    "page_not_processed": "high",
    "export_inconsistency": "high",
    "suspected_ocr_error": "medium",
    "low_ocr_confidence": "medium",
    "text_overflow": "medium",
    "visual_text_leak": "medium",
    "inpaint_artifact": "medium",
    "mask_missing": "medium",
    "duplicate_log_event": "low",
}

RENDER_POLICY = {
    "approved": "render",
    "warning": "render_with_flag",
    "blocked": "debug_overlay_only",
    "ignored": "skip_with_reason",
}


def severity_for_flag(flag: str) -> str:
    return FLAG_SEVERITY.get(flag, "low")


def dedupe_flags(flags: list[str]) -> list[str]:
    seen = set()
    result = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        result.append(flag)
    return result


def summarize_flags(regions: list[dict[str, Any]]) -> dict[str, Any]:
    all_flags: list[str] = []
    for region in regions:
        all_flags.extend(region.get("qa_flags") or [])
    deduped = dedupe_flags(all_flags)
    counts = Counter(all_flags)
    highest = "low"
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    for flag in deduped:
        severity = severity_for_flag(flag)
        if rank[severity] > rank[highest]:
            highest = severity
    return {
        "flags": deduped,
        "counts": dict(counts),
        "highest_severity": highest if deduped else "none",
        "critical_count": sum(count for flag, count in counts.items() if severity_for_flag(flag) == "critical"),
        "total": len(all_flags),
    }


def render_decision(region: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    status = region.get("review_status") or "approved"
    flags = dedupe_flags(region.get("qa_flags") or [])
    if any(severity_for_flag(flag) == "critical" for flag in flags):
        status = "blocked"
    elif flags and status == "approved":
        status = "warning"
    if status == "blocked" and debug:
        action = "debug_overlay_only"
    else:
        action = RENDER_POLICY.get(status, "render")
    return {"status": status, "action": action, "flags": flags}


def can_export(summary: dict[str, Any], mode: str = "strict") -> dict[str, Any]:
    critical = int(summary.get("critical_count", 0) or 0)
    if mode == "strict" and critical:
        return {"allowed": False, "reason": "critical_flags_block_strict_export"}
    if mode == "review" and critical:
        return {"allowed": True, "reason": "review_export_requires_debug_or_warning"}
    if mode == "unsafe/manual":
        return {"allowed": True, "reason": "manual_override_record_required"}
    return {"allowed": True, "reason": "ok"}

