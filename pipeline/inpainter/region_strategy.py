"""Select inpaint strategy by region type."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inpainter.mask_validator import validate_mask
from sfx.inpaint_gate import evaluate_sfx_inpaint_gate


REGION_STRATEGY = {
    "text": "component_roi_snap8",
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

TEXT_REGION_STRATEGY = REGION_STRATEGY["text"]
ROI_STRATEGY_KEYS = {"text"}


def classify_region(region: dict[str, Any]) -> str:
    background_type = str(region.get("background_type") or "").strip()
    if background_type == "sfx_text":
        return "sfx_text"
    if background_type in REGION_STRATEGY:
        return background_type
    if region.get("tipo") == "narracao":
        return "caption_box"
    return "text"


def plan_inpaint(region: dict[str, Any], mask_path: str | Path | None, *, allow_sfx: bool = False) -> dict[str, Any]:
    region_type = classify_region(region)
    if region_type == "sfx_text" and not allow_sfx:
        return {"run": False, "strategy": REGION_STRATEGY[region_type], "qa_flags": ["sfx_preserved"], "region_type": region_type}
    sfx_gate: dict[str, Any] | None = None
    if region_type == "sfx_text":
        sfx_gate = evaluate_sfx_inpaint_gate(region)
        if not sfx_gate["allow_inpaint"]:
            return {
                "run": False,
                "strategy": sfx_gate["strategy"],
                "qa_flags": sfx_gate["qa_flags"],
                "region_type": region_type,
                "sfx_inpaint_gate": sfx_gate,
            }
    if not mask_path:
        return {"run": False, "strategy": "blocked", "qa_flags": ["mask_missing"], "region_type": region_type}
    mask = validate_mask(mask_path, region.get("bbox"))
    if not mask["valid"]:
        return {"run": False, "strategy": "blocked", "qa_flags": [mask["reason"]], "region_type": region_type}
    return {
        "run": True,
        "strategy": sfx_gate["strategy"] if sfx_gate else REGION_STRATEGY.get(region_type, TEXT_REGION_STRATEGY),
        "qa_flags": sfx_gate["qa_flags"] if sfx_gate else [],
        "region_type": region_type,
        "roi_strategy": "component_roi_snap8" if region_type in ROI_STRATEGY_KEYS else ("component_roi_snap8" if sfx_gate else None),
        **({"sfx_inpaint_gate": sfx_gate} if sfx_gate else {}),
    }


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

