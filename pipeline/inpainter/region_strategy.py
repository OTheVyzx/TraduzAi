"""Select inpaint strategy by region type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inpainter.mask_validator import validate_mask


REGION_STRATEGY = {
    "white_balloon": "telea_fast",
    "colored_balloon": "lama_or_patchmatch",
    "caption_box": "preserve_borders",
    "solid_background": "telea_fast",
    "textured_background": "lama_required",
    "gradient_background": "lama_required",
    "dark_background": "lama_required",
    "sfx_text": "skip_without_explicit_config",
    "map_texture": "manual_review",
    "sky_texture": "lama_required",
    "character_overlap": "manual_review",
}


def classify_region(region: dict[str, Any]) -> str:
    if region.get("tipo") == "sfx":
        return "sfx_text"
    if region.get("tipo") == "narracao":
        return "caption_box"
    return region.get("background_type") or region.get("balloon_type") or "white_balloon"


def plan_inpaint(region: dict[str, Any], mask_path: str | Path | None, *, allow_sfx: bool = False) -> dict[str, Any]:
    region_type = classify_region(region)
    if region_type == "sfx_text" and not allow_sfx:
        return {"run": False, "strategy": REGION_STRATEGY[region_type], "qa_flags": ["sfx_preserved"], "region_type": region_type}
    if not mask_path:
        return {"run": False, "strategy": "blocked", "qa_flags": ["mask_missing"], "region_type": region_type}
    mask = validate_mask(mask_path, region.get("bbox"))
    if not mask["valid"]:
        return {"run": False, "strategy": "blocked", "qa_flags": [mask["reason"]], "region_type": region_type}
    return {"run": True, "strategy": REGION_STRATEGY.get(region_type, "manual_review"), "qa_flags": [], "region_type": region_type}


def debug_output_paths(debug_root: str | Path, page: int) -> dict[str, Path]:
    root = Path(debug_root) / "inpaint"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"page_{page:03}"
    return {
        "before": root / f"{stem}_before.png",
        "mask": root / f"{stem}_mask.png",
        "after": root / f"{stem}_after.png",
        "diff": root / f"{stem}_diff.png",
    }

