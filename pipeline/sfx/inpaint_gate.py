"""Conservative inpaint gate for manhwa SFX candidates."""

from __future__ import annotations

from typing import Any


BLOCKING_QA_FLAGS = {
    "sfx_overlaps_character_art",
    "character_overlap",
    "complex_background",
    "sfx_inpaint_damaged_art_risk",
    "mask_density_high",
    "sfx_mask_density_high",
}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _qa_flags(region: dict[str, Any]) -> list[str]:
    flags = [str(flag) for flag in region.get("qa_flags") or [] if flag]
    sfx = region.get("sfx") if isinstance(region.get("sfx"), dict) else {}
    flags.extend(str(flag) for flag in sfx.get("qa_flags") or [] if flag)
    evidence = region.get("mask_evidence") if isinstance(region.get("mask_evidence"), dict) else {}
    flags.extend(str(flag) for flag in evidence.get("qa_flags") or [] if flag)
    return list(dict.fromkeys(flags))


def _manual_override(region: dict[str, Any]) -> bool:
    sfx = region.get("sfx") if isinstance(region.get("sfx"), dict) else {}
    return bool(
        region.get("manual_sfx_inpaint_override")
        or region.get("allow_sfx_inpaint_override")
        or sfx.get("manual_override")
    )


def evaluate_sfx_inpaint_gate(region: dict[str, Any]) -> dict[str, Any]:
    """Return whether a SFX candidate is safe enough for automatic inpaint."""

    flags = _qa_flags(region)
    evidence = region.get("mask_evidence") if isinstance(region.get("mask_evidence"), dict) else {}
    if _manual_override(region):
        return {
            "allow_inpaint": True,
            "strategy": "lama_component_roi",
            "qa_flags": list(dict.fromkeys([*flags, "manual_override"])),
            "reason": "manual_override",
        }

    if evidence.get("kind") != "sfx_glyph_mask":
        return {
            "allow_inpaint": False,
            "strategy": "review_required",
            "qa_flags": list(dict.fromkeys([*flags, "sfx_mask_evidence_missing"])),
            "reason": "missing_sfx_glyph_mask",
        }

    blocking_flags = sorted(set(flags) & BLOCKING_QA_FLAGS)
    if blocking_flags:
        return {
            "allow_inpaint": False,
            "strategy": "review_required",
            "qa_flags": list(dict.fromkeys([*flags, *blocking_flags])),
            "reason": blocking_flags[0],
        }

    fill_ratio = _as_float(evidence.get("bbox_fill_ratio"))
    component_count = _as_float(evidence.get("component_count"))
    moderate_componentized_mask = (
        fill_ratio is not None
        and 0.34 < fill_ratio <= 0.45
        and component_count is not None
        and component_count >= 2
        and "touches_most_crop_border" not in flags
    )
    if fill_ratio is not None and fill_ratio > 0.34 and not moderate_componentized_mask:
        return {
            "allow_inpaint": False,
            "strategy": "review_required",
            "qa_flags": list(dict.fromkeys([*flags, "sfx_mask_density_high"])),
            "reason": "sfx_mask_density_high",
        }

    expanded_pixels = _as_float(evidence.get("expanded_mask_pixels"))
    if expanded_pixels is not None and expanded_pixels <= 0:
        return {
            "allow_inpaint": False,
            "strategy": "review_required",
            "qa_flags": list(dict.fromkeys([*flags, "sfx_mask_empty"])),
            "reason": "sfx_mask_empty",
        }

    return {
        "allow_inpaint": True,
        "strategy": "lama_component_roi",
        "qa_flags": flags,
        "reason": "safe_sfx_glyph_mask",
    }
