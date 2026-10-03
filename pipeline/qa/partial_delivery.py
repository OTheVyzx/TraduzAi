"""Auditable partial delivery; never grants pixel or ownership authority."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from qa.gate_composition import normalize_export_gate
from translator.delivery_policy import POLICY_ID

# These findings describe incomplete/uncertain quality, not corrupt evidence.
_WARNING_REASONS = frozenset({
    "independently_detected_text_without_owner",
    "uncertain_source_component_requires_review",
    "terminal_source_language_residual",
    "style_fidelity_high_confidence_mismatch",
    "style_fidelity_coverage_missing",
    "unchanged_source_dialogue", "source_script_leak", "target_language_uncertain",
})


def notice_for_layer(layer, *, binding=None, attempts=(), commits=()):
    """Use verified page-result identities, never geometry guesses, to report use."""
    owner_id = layer.get("owner_id") or layer.get("candidate_owner_id") or layer.get("id")
    attempt_rows = [a for a in attempts if a.owner_id == owner_id]
    matched = [c for c in commits if binding is not None
               and c.owner_id == binding.owner_id
               and c.translation_binding_sha256 == binding.translation_binding_sha256
               and c.source_payload_sha256 == binding.source_payload_sha256
               and c.target_payload_sha256 == binding.target_payload_sha256]
    applied = bool(matched and layer.get("owner_id") == owner_id
                   and not layer.get("candidate_owner_id") and layer.get("visible") is True)
    preserved = bool(binding is not None and binding.preserves_original_pixels)
    target = binding.target_text if binding is not None else None
    if binding is None:
        for attempt in reversed(attempt_rows):
            metadata = json.loads(attempt.provider_metadata_json_bytes)
            if "target_produced" in metadata:
                target = metadata["target_produced"]
                break
    reasons = []
    if binding is not None and binding.quality_warning_reason:
        reasons.append(binding.quality_warning_reason)
    if not applied and not preserved:
        reasons.append(str(layer.get("owner_execution_rejection_reason")
                           or layer.get("skip_reason") or "insertion_not_committed"))
    done = ["source_recorded"]
    if any(a.provider_called or a.cache_hit for a in attempt_rows):
        done.append("translation_provider_called_or_cache_verified")
    if binding is not None:
        done.extend(["translation_bound", "language_validation"])
    if applied:
        done.extend(["cleanup_committed", "insertion_committed"])
    if preserved:
        done.append("source_preserved_by_explicit_policy")
    pending = [] if applied or preserved else ["safe_owner_geometry_and_insertion"]
    if reasons and binding is not None and binding.quality_warning_reason:
        pending.append("linguistic_review")
    return {
        "policy_id": POLICY_ID, "page_id": layer.get("page_id"), "owner_id": owner_id,
        "original": binding.source_text if binding else str(layer.get("source_payload") or layer.get("original") or layer.get("text") or ""),
        "target_produced": target, "target_used": target if applied else None,
        "target_recording_status": "bound_response" if binding else "unbound_response" if target is not None else "no_recorded_response",
        "insertion_applied": applied, "source_preserved_by_policy": preserved,
        "status": "applied_with_warnings" if applied and reasons else "applied" if applied
                  else "preserved_by_policy" if preserved else "not_applied",
        "warning_reasons": reasons, "steps_done": done,
        "steps_not_done": [] if applied or preserved else ["cleanup", "insertion"],
        "pending_work": pending,
        "commit_ids": [c.commit_id for c in matched] if applied else [],
        "translation_binding_sha256": binding.translation_binding_sha256 if binding else None,
        "attempt_ids": [a.attempt_id for a in attempt_rows],
    }


def apply_partial_delivery_policy(project: dict[str, Any], gate: dict[str, Any]):
    """Keep the complete original gate and demote only authorized quality reasons."""
    qa = project.setdefault("qa", {})
    qa["original_export_gate"] = deepcopy(gate)
    issues = deepcopy(gate.get("issues") or [])
    for issue in issues:
        reason = str(issue.get("reason") or issue.get("code") or "")
        flags = {str(flag) for flag in issue.get("flags") or []}
        if reason in _WARNING_REASONS and flags.issubset(_WARNING_REASONS | {reason}):
            issue["original_severity"] = issue.get("severity")
            issue["original_blocks_export"] = issue.get("blocks_export")
            issue.update(severity="warning", blocks_export=False,
                         delivery_policy_id=POLICY_ID, quality_approved=False)
    notices = []
    for page in project.get("paginas") or project.get("pages") or []:
        for layer in page.get("text_layers") or page.get("textos") or []:
            notice = layer.get("translation_delivery_notice")
            if not isinstance(notice, dict):
                notice = notice_for_layer(layer)
            notices.append(deepcopy(notice))
            if notice.get("warning_reasons"):
                issues.append({"type": "translation_delivery_notice", "reason": "translation_delivery_requires_review",
                               "severity": "warning", "blocks_export": False,
                               "owner_id": notice.get("owner_id"), "page_id": notice.get("page_id"),
                               "flags": list(notice["warning_reasons"]), "delivery_policy_id": POLICY_ID,
                               "quality_approved": False})
    warnings = any(i.get("severity") == "warning" for i in issues)
    partial = any(n.get("status") == "not_applied" for n in notices)
    # A BLOCK without a canonical issue is an integrity failure, not quality.
    missing_block_evidence = gate.get("status") == "BLOCK" and not issues
    result = normalize_export_gate(issues, blocked=missing_block_evidence,
                                   review=warnings or partial, override=False)
    report = {"policy_id": POLICY_ID, "items": notices,
              "applied_count": sum(n["insertion_applied"] for n in notices),
              "not_applied_count": sum(n["status"] == "not_applied" for n in notices),
              "quality_approved": result["status"] == "PASS" and not partial,
              "status": "blocked_integrity" if result["status"] == "BLOCK" else
                        "partial_with_warnings" if partial else
                        "completed_with_warnings" if warnings else "completed"}
    qa["translation_delivery"] = report
    project["translation_delivery_status"] = report["status"]
    project["needs_review"] = bool(result["needs_review"] or partial)
    return result
