"""Translation QA flags, severity and blocking policies."""

from __future__ import annotations

from collections import Counter
from typing import Any

FLAG_SEVERITY = {
    "glossary_violation": "critical",
    "forbidden_translation": "critical",
    "entity_mistranslated": "critical",
    "placeholder_lost": "critical",
    "unrestored_placeholder": "critical",
    "schema_validation_error": "critical",
    "vlm_failure_phrase": "critical",
    "translation_fallback_phrase": "critical",
    "source_script_leak": "critical",
    "speech_cjk_preserved_inside_balloon": "critical",
    "mask_outside_balloon_critical": "critical",
    "bbox_overreach_critical": "critical",
    "layout_bbox_coordinate_mismatch": "critical",
    "bubble_inner_bbox_coordinate_mismatch": "critical",
    "page_space_rerender_mixed_coordinates": "critical",
    "source_bbox_assigned_from_balloon": "critical",
    "render_outside_balloon": "critical",
    "render_bbox_far_from_target_bbox": "critical",
    "render_on_art_suspected": "critical",
    "fit_below_minimum_legible": "critical",
    "missing_render_bbox": "critical",
    "text_residual_after_inpaint": "critical",
    "text_residual_after_inpaint_confirmed": "critical",
    "fast_fill_unverified_residual": "critical",
    "fast_fill_insufficient_coverage": "critical",
    "fast_fill_no_glyph_evidence": "critical",
    "special_class_rendered_as_dialogue": "critical",
    "mojibake_in_translation": "critical",
    "skip_inpaint_not_honored": "critical",
    "strict_gate_not_enforced": "critical",
    "qa_summary_and_export_gate_diverge": "critical",
    "qa_flag_not_propagated": "critical",
    "empty_translation": "high",
    "untranslated_english": "high",
    "gibberish_detected": "high",
    "page_not_processed": "high",
    "export_inconsistency": "high",
    "text_overflow_high": "high",
    "outline_damage_high": "high",
    "TEXT_CLIPPED": "high",
    "TEXT_OVERFLOW": "high",
    "ocr_truncated_or_joined": "high",
    "ocr_partial_low_confidence_fragment": "high",
    "bbox_overreach": "high",
    "mask_outside_balloon": "high",
    "weak_text_residual_after_inpaint": "high",
    "text_residual_after_inpaint_suspected": "high",
    "render_outside_balloon_suspected": "high",
    "safe_text_box_recomputed": "high",
    "low_inpaint_coverage": "high",
    "balloon_bbox_collapsed_to_text": "high",
    "balloon_bbox_missing": "high",
    "lobe_assignment_low_confidence": "high",
    "rotated_text_policy_unmet": "high",
    "suspected_ocr_error": "medium",
    "literal_ocr_translation": "medium",
    "low_ocr_confidence": "medium",
    "text_overflow": "medium",
    "visual_text_leak": "medium",
    "inpaint_artifact": "medium",
    "mask_missing": "medium",
    "safe_text_box_outside_balloon": "medium",
    "sign_render_outside_region": "medium",
    "tn_note_rendered_as_speech": "medium",
    "url_watermark_inpainted": "medium",
    "ocr_run_on_suspect": "medium",
    "ocr_false_positive_review": "medium",
    "ocr_duplicate_garble_review": "medium",
    "duplicate_log_event": "low",
    "top_narration": "low",
}
REMOVED_VISUAL_FILTER_FLAGS = {
    "low_confidence_visual_noise",
    "cover_title_logo",
    "mask_density_high",
}

RENDER_POLICY = {
    "approved": "render",
    "warning": "render_with_flag",
    "blocked": "debug_overlay_only",
    "ignored": "skip_with_reason",
}


def severity_for_flag(flag: str) -> str:
    if flag in REMOVED_VISUAL_FILTER_FLAGS:
        return "low"
    return FLAG_SEVERITY.get(flag, "low")


def dedupe_flags(flags: list[str]) -> list[str]:
    seen = set()
    result = []
    for flag in flags:
        if flag in REMOVED_VISUAL_FILTER_FLAGS:
            continue
        if flag in seen:
            continue
        seen.add(flag)
        result.append(flag)
    return result


def summarize_flags(regions: list[dict[str, Any]]) -> dict[str, Any]:
    all_flags: list[str] = []
    issue_count = 0
    critical_issue_count = 0
    warning_issue_count = 0
    for region in regions:
        flags = [
            str(flag)
            for flag in region.get("qa_flags") or []
            if str(flag) not in REMOVED_VISUAL_FILTER_FLAGS
        ]
        if flags:
            issue_count += 1
        if any(severity_for_flag(flag) == "critical" for flag in flags):
            critical_issue_count += 1
        if any(severity_for_flag(flag) == "high" for flag in flags):
            warning_issue_count += 1
        all_flags.extend(flags)
    deduped = dedupe_flags(all_flags)
    counts = Counter(all_flags)
    highest = "low"
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    for flag in deduped:
        severity = severity_for_flag(flag)
        if rank[severity] > rank[highest]:
            highest = severity
    critical_flag_count = sum(count for flag, count in counts.items() if severity_for_flag(flag) == "critical")
    warning_flag_count = sum(count for flag, count in counts.items() if severity_for_flag(flag) == "high")
    return {
        "flags": deduped,
        "counts": dict(counts),
        "highest_severity": highest if deduped else "none",
        "issue_count": issue_count,
        "critical_issue_count": critical_issue_count,
        "critical_flag_count": critical_flag_count,
        "warning_issue_count": warning_issue_count,
        "warning_flag_count": warning_flag_count,
        "critical_count": critical_flag_count,
        "total": len(all_flags),
    }


def render_decision(region: dict[str, Any], *, debug: bool = False) -> dict[str, Any]:
    status = region.get("review_status") or "approved"
    flags = dedupe_flags([str(flag) for flag in region.get("qa_flags") or []])
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

