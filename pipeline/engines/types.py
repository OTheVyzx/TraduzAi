from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

PipelineQuality = Literal["normal", "ultra"]


class DetectionEngine(Protocol):
    name: str

    def detect(self, image_rgb: Any, conf_threshold: float | None = None) -> list[Any]:
        ...


class TextRefinerEngine(Protocol):
    name: str

    def refine(self, image_rgb: Any, blocks: list[Any], *, quality: PipelineQuality = "normal") -> list[dict]:
        ...


class MaskEngine(Protocol):
    name: str


class InpaintEngine(Protocol):
    name: str


class ValidationEngine(Protocol):
    name: str


@dataclass
class EngineBundle:
    detector: DetectionEngine
    text_refiner: TextRefinerEngine
    mask_engine: MaskEngine
    inpaint_engine: InpaintEngine
    validator: ValidationEngine
    quality: PipelineQuality
    ocr_profile: str = "max"
    detector_name: str = "legacy_visual_stack"
    text_refiner_name: str = "legacy_ocr_stage"
    mask_name: str = "legacy_vision_blocks"
    inpaint_name: str = "legacy_inpaint_band"
    validator_name: str = "off"

    def summary(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "detector": self.detector_name,
            "text_refiner": self.text_refiner_name,
            "mask": self.mask_name,
            "inpaint": self.inpaint_name,
            "validation": self.validator_name,
            "ocr_profile": self.ocr_profile,
            "fallbacks": [],
        }
