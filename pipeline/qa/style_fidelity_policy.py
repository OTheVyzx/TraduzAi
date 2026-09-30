"""Fail-closed policy constants for the owner-scoped style fidelity gate."""

from __future__ import annotations

from typing import Any


HIGH_CONFIDENCE_THRESHOLD = 0.80
RENDERED_OWNER_STATES = frozenset({"rendered"})
REQUIRED_RENDER_METRICS = frozenset(
    {
        "core_pixel_count",
        "effect_pixel_count",
        "core_pixels_outside_safe",
        "effect_pixels_outside_safe",
    }
)
POLICY_ADJUSTMENT_WHITELIST = frozenset(
    {"font_size_px", "tracking_xh", "alignment", "scale_y", "width_scale"}
)
MINIMUM_LEGIBLE_FONT_SIZE_PX = 8.0
CATASTROPHIC_STYLE_ATTRIBUTES = frozenset(
    {
        "font_name",
        "fill",
        "stroke",
        "gradient",
        "glow",
        "shadow",
        "outline",
    }
)


def required_style_categories(project: dict[str, Any]) -> tuple[str, ...]:
    """Return explicit required categories without inventing absent content classes."""

    configured = project.get("style_fidelity_required_categories")
    if not isinstance(configured, list):
        return ("all_renderable",)
    categories = tuple(sorted({str(value).strip() for value in configured if str(value).strip()}))
    return categories or ("all_renderable",)


def is_rendered_owner(owner: Any) -> bool:
    """Whether an owner belongs in the immutable fidelity denominator."""

    state = str(getattr(owner, "state", "") or "").strip().lower()
    disposition = str(getattr(owner, "disposition", "") or "").strip().lower()
    return disposition == "owned" and state in RENDERED_OWNER_STATES
