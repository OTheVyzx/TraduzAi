from __future__ import annotations

import logging
import os
from typing import Any

from .types import EngineBundle, PipelineQuality

logger = logging.getLogger(__name__)


def normalize_pipeline_quality(value: Any) -> PipelineQuality:
    if isinstance(value, dict):
        raw = value.get("pipeline_quality") or value.get("qualidade") or "normal"
    else:
        raw = value
    normalized = str(raw or "").strip().lower()
    if normalized in {"ultra", "alta", "max", "maximum"}:
        return "ultra"
    return "normal"


def _engine_flag(name: str, allowed: set[str], default: str = "auto") -> str:
    raw = os.getenv(name, default)
    value = str(raw or default).strip().lower()
    if value not in allowed:
        logger.warning("Flag %s=%s invalida; usando %s", name, raw, default)
        return default
    return value


class LegacyDetectorEngine:
    name = "legacy_visual_stack"

    def __init__(self, profile: str = "max") -> None:
        self.profile = profile

    def detect(self, image_rgb: Any, conf_threshold: float | None = None) -> list[Any]:
        from vision_stack.runtime import _get_detector, _profile_to_detection_threshold

        threshold = conf_threshold if conf_threshold is not None else _profile_to_detection_threshold(self.profile)
        return _get_detector(self.profile).detect(image_rgb, conf_threshold=threshold)


class LegacyTextRefinerEngine:
    name = "legacy_ocr_stage"

    def refine(self, image_rgb: Any, blocks: list[Any], *, quality: PipelineQuality = "normal") -> list[dict]:
        del image_rgb, quality
        refined = []
        for block in blocks:
            bbox = list(getattr(block, "xyxy", (0, 0, 0, 0)))
            refined.append(
                {
                    "bbox": bbox,
                    "source_bbox": bbox,
                    "line_polygons": list(getattr(block, "line_polygons", []) or []),
                    "text_pixel_bbox": bbox,
                    "confidence": getattr(block, "confidence", 1.0),
                    "text_refiner": self.name,
                }
            )
        return refined


class LegacyMaskEngine:
    name = "legacy_vision_blocks"


class LegacyInpaintEngine:
    name = "legacy_inpaint_band"

    def inpaint_band_image(self, band_rgb: Any, ocr_page: dict) -> Any:
        from inpainter import inpaint_band_image

        return inpaint_band_image(band_rgb, ocr_page)


class DisabledValidationEngine:
    name = "off"

    def validate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "status": "off",
            "residual_bboxes": [],
            "retry_recommended": False,
            "confidence": 0.0,
            "engine": self.name,
        }


def _build_detector_engine(config: dict[str, Any], quality: PipelineQuality):
    flag = _engine_flag("TRADUZAI_ENGINE_DETECTOR", {"auto", "legacy", "comic_layout_rtdetr"})
    legacy = LegacyDetectorEngine(profile="max")
    if flag == "comic_layout_rtdetr":
        from vision_stack.comic_layout_detector import ComicLayoutRTDetrDetector

        return ComicLayoutRTDetrDetector(
            models_dir=config.get("models_dir"),
            quality=quality,
            fallback=legacy,
        )
    return legacy


def _build_text_refiner_engine(config: dict[str, Any], quality: PipelineQuality):
    flag = _engine_flag("TRADUZAI_ENGINE_REFINER", {"auto", "legacy", "ppocrv5"})
    if flag == "ppocrv5":
        from vision_stack.text_refiner import PPOCRv5TextRefiner

        return PPOCRv5TextRefiner(
            quality=quality,
            lang=str(config.get("idioma_origem") or "en"),
        )
    return LegacyTextRefinerEngine()


def _build_mask_engine(config: dict[str, Any], quality: PipelineQuality):
    del config, quality
    flag = _engine_flag("TRADUZAI_ENGINE_MASK", {"auto", "legacy", "smart"})
    if flag == "smart":
        from vision_stack.smart_text_mask import SmartTextMaskEngine

        return SmartTextMaskEngine()
    return LegacyMaskEngine()


def _build_validation_engine(config: dict[str, Any], quality: PipelineQuality):
    del config
    flag = _engine_flag("TRADUZAI_ENGINE_VALIDATION", {"auto", "off", "residual"}, default="off")
    if flag == "residual":
        from vision_stack.residual_validator import ResidualValidationEngine

        return ResidualValidationEngine(quality=quality)
    return DisabledValidationEngine()


def _build_inpaint_engine(config: dict[str, Any], quality: PipelineQuality, validator: Any):
    flag = _engine_flag("TRADUZAI_ENGINE_INPAINT", {"auto", "legacy", "lama_onnx"})
    if flag == "lama_onnx":
        from inpainter.engines import CompositeBandInpaintEngine

        return CompositeBandInpaintEngine(
            quality=quality,
            models_dir=config.get("models_dir"),
            validator=validator,
        )
    return LegacyInpaintEngine()


def resolve_engines(config: dict[str, Any]) -> EngineBundle:
    quality = normalize_pipeline_quality(config)
    detector = _build_detector_engine(config, quality)
    detector_name = getattr(detector, "name", "legacy_visual_stack")
    text_refiner = _build_text_refiner_engine(config, quality)
    text_refiner_name = getattr(text_refiner, "name", "legacy_ocr_stage")
    mask_engine = _build_mask_engine(config, quality)
    mask_name = getattr(mask_engine, "name", "legacy_vision_blocks")
    validator = _build_validation_engine(config, quality)
    validator_name = getattr(validator, "name", "off")
    inpaint_engine = _build_inpaint_engine(config, quality, validator)
    inpaint_name = getattr(inpaint_engine, "name", "legacy_inpaint_band")

    bundle = EngineBundle(
        detector=detector,
        text_refiner=text_refiner,
        mask_engine=mask_engine,
        inpaint_engine=inpaint_engine,
        validator=DisabledValidationEngine(),
        quality=quality,
        ocr_profile="max",
        detector_name=detector_name,
        text_refiner_name=text_refiner_name,
        mask_name=mask_name,
        inpaint_name=inpaint_name,
        validator_name=validator_name,
    )
    logger.info("Engine bundle resolvido: %s", bundle.summary())
    return bundle
