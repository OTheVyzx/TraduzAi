"""Independent observation of text visible in persisted final-page bytes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class FinalPixelObservation:
    image_path: Path
    persisted_sha256: str
    image_rgb: np.ndarray
    detected_blocks: tuple[dict[str, Any], ...]
    ocr_records: tuple[dict[str, Any], ...]
    source_language: str

    def __post_init__(self) -> None:
        image = np.ascontiguousarray(self.image_rgb, dtype=np.uint8).copy()
        image.setflags(write=False)
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(
            self,
            "detected_blocks",
            tuple(copy.deepcopy(dict(block)) for block in self.detected_blocks),
        )
        object.__setattr__(
            self,
            "ocr_records",
            tuple(copy.deepcopy(dict(record)) for record in self.ocr_records),
        )


class FinalPixelObserver(Protocol):
    def observe(
        self,
        image_path: Path,
        *,
        source_language: str,
    ) -> FinalPixelObservation: ...


def _detector_block(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    bbox = None
    if all(hasattr(value, field) for field in ("x1", "y1", "x2", "y2")):
        bbox = [
            int(round(float(value.x1))),
            int(round(float(value.y1))),
            int(round(float(value.x2))),
            int(round(float(value.y2))),
        ]
    if bbox is None:
        raise ValueError("fresh detector returned a block without canonical bbox evidence")
    block = {"bbox": bbox}
    confidence = getattr(value, "confidence", None)
    if confidence is not None:
        block["confidence"] = float(confidence)
    return block


class DetectorOcrFinalPixelObserver:
    """Read one persisted file and run a fresh full-page detector/OCR pass."""

    def __init__(self, *, detector: Any, runtime: Any) -> None:
        self._detector = detector
        self._runtime = runtime

    def observe(
        self,
        image_path: Path,
        *,
        source_language: str,
    ) -> FinalPixelObservation:
        path = Path(image_path)
        payload = path.read_bytes()
        persisted_sha256 = sha256(payload).hexdigest()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"persisted final page is not a decodable RGB image: {path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        persisted_image_rgb = image_rgb.copy()

        detect = getattr(self._detector, "detect", None)
        if not callable(detect):
            raise TypeError("final pixel detector has no callable detect method")
        raw_blocks = detect(persisted_image_rgb.copy())
        if hasattr(raw_blocks, "blocks"):
            raw_blocks = raw_blocks.blocks
        if raw_blocks is None:
            raw_blocks = []
        if isinstance(raw_blocks, (str, bytes, dict)):
            raise ValueError("fresh detector returned a non-sequence result")
        detected_blocks = tuple(_detector_block(block) for block in raw_blocks)

        run_ocr_stage = getattr(self._runtime, "run_ocr_stage", None)
        if not callable(run_ocr_stage):
            raise TypeError("final pixel OCR runtime has no callable run_ocr_stage method")
        page = {
            "numero": 1,
            "width": int(image_rgb.shape[1]),
            "height": int(image_rgb.shape[0]),
            "source_language": str(source_language),
            "_vision_blocks": [copy.deepcopy(block) for block in detected_blocks],
            "_final_pixel_fresh_observation": True,
        }
        ocr_result = run_ocr_stage(persisted_image_rgb.copy(), page)
        if not isinstance(ocr_result, dict):
            raise ValueError("fresh final pixel OCR returned a non-mapping result")
        raw_records = ocr_result.get("texts") or []
        if not isinstance(raw_records, list) or any(
            not isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("fresh final pixel OCR returned malformed text records")
        return FinalPixelObservation(
            image_path=path,
            persisted_sha256=persisted_sha256,
            image_rgb=persisted_image_rgb,
            detected_blocks=detected_blocks,
            ocr_records=tuple(raw_records),
            source_language=str(source_language),
        )
