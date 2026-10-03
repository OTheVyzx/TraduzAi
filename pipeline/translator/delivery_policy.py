"""Explicit partial-delivery policy; quality findings remain visible.

This policy never supplies missing text, geometry, masks or owner authority.
"""
POLICY_ID = "available_translation_with_warnings_v1"

def usable_with_warning(target, verdict):
    return bool(str(target or "").strip() and verdict is not None
                and not verdict.accepted and verdict.reason not in {
                    "empty_target", "placeholder_mismatch", "entity_mismatch"})

def quality_warning_metadata(target, verdict):
    if not usable_with_warning(target, verdict):
        raise ValueError("No usable translation response for warning delivery")
    return {"quality_usage_policy_id":POLICY_ID,"quality_warning_reason":verdict.reason,
            "quality_approved":False,"validation_performed":True}
