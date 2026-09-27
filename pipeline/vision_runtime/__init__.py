"""Public, lazily loaded Vision Runtime V1 API for Integration consumers."""

from __future__ import annotations

from importlib import import_module
from typing import Any


VISION_RUNTIME_VERSION = "1.0.0"

_EXPORTS = {
    "AnalysisArtifactCache": ("analysis_cache", "AnalysisArtifactCache"),
    "AnalysisCacheIdentity": ("analysis_cache", "AnalysisCacheIdentity"),
    "AuthenticatedContextWindow": ("context_window", "AuthenticatedContextWindow"),
    "assess_observation_evidence": (
        "observation_evidence", "assess_observation_evidence"
    ),
    "build_analysis_payload": ("analysis_payload", "build_analysis_payload"),
    "build_structural_analysis": ("structure", "build_structural_analysis"),
    "discover_white_containers": ("white_containers", "discover_white_containers"),
    "plan_vertical_refinement": ("ocr_refinement", "plan_vertical_refinement"),
    "polygon_mask": ("raster_authority", "polygon_mask"),
    "preserved_art_mask": ("raster_authority", "preserved_art_mask"),
    "reconcile_observations": ("ocr_refinement", "reconcile_observations"),
    "select_ocr_invocation": ("ocr_selection", "select_ocr_invocation"),
}


def capability_manifest() -> dict[str, Any]:
    """Return stable capability names without loading heavy optional providers."""

    return {
        "schema": "traduzai.vision-runtime.v1",
        "version": VISION_RUNTIME_VERSION,
        "capabilities": {
            "analysis_payload": "v1",
            "dependency_cache": "v1",
            "ocr_consensus": "v1",
            "ocr_refinement": "v1",
            "observation_evidence": "v1",
            "structure": "v1",
            "authenticated_context": "v1",
            "raster_authority": "v1",
        },
        "mask_channels": [
            "glyph", "outline", "shadow", "glow", "ignore_or_uncertain"
        ],
    }


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


__all__ = ["VISION_RUNTIME_VERSION", "capability_manifest", *_EXPORTS]

