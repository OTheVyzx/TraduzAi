"""Independent observation of text visible in persisted final-page bytes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, Sequence

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
    page_id: str = ""
    page_number: int = 0
    detected_block_count: int = 0
    ocr_record_count: int = 0
    ocr_attempts: tuple[dict[str, Any], ...] = ()
    expected_source_challenge_count: int = 0
    completed_source_challenge_count: int = 0
    coverage_complete: bool = True
    coverage_failures: tuple[str, ...] = ()

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
        object.__setattr__(
            self,
            "ocr_attempts",
            tuple(copy.deepcopy(dict(attempt)) for attempt in self.ocr_attempts),
        )
        object.__setattr__(
            self,
            "coverage_failures",
            tuple(sorted(set(str(value) for value in self.coverage_failures))),
        )


class FinalPixelObserver(Protocol):
    def observe(
        self,
        image_path: Path,
        *,
        source_language: str,
        page_id: str = "",
        page_number: int = 0,
        source_challenges: Sequence[dict[str, Any]] = (),
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
        page_id: str = "",
        page_number: int = 0,
        source_challenges: Sequence[dict[str, Any]] = (),
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

        run_probe = getattr(self._runtime, "run_final_pixel_ocr_probe", None)
        if not callable(run_probe):
            raise TypeError(
                "final pixel OCR runtime has no callable run_final_pixel_ocr_probe method"
            )
        probe = run_probe(
            persisted_image_rgb.copy(),
            detected_blocks=[copy.deepcopy(block) for block in detected_blocks],
            source_challenges=[copy.deepcopy(item) for item in source_challenges],
            page_id=str(page_id or ""),
            page_number=int(page_number or 0),
            source_language=str(source_language),
        )

        def probe_field(name: str, default: Any) -> Any:
            return probe.get(name, default) if isinstance(probe, dict) else getattr(probe, name, default)

        raw_records = probe_field("raw_ocr_records", ())
        if not isinstance(raw_records, (list, tuple)) or any(
            not isinstance(record, dict) for record in raw_records
        ):
            raise ValueError("fresh final pixel OCR returned malformed text records")
        attempts = probe_field("ocr_attempts", ())
        failures = probe_field("coverage_failures", ())
        return FinalPixelObservation(
            image_path=path,
            persisted_sha256=persisted_sha256,
            image_rgb=persisted_image_rgb,
            detected_blocks=detected_blocks,
            ocr_records=tuple(raw_records),
            source_language=str(source_language),
            page_id=str(page_id or ""),
            page_number=int(page_number or 0),
            detected_block_count=len(detected_blocks),
            ocr_record_count=len(raw_records),
            ocr_attempts=tuple(attempts),
            expected_source_challenge_count=int(
                probe_field("expected_source_challenge_count", len(source_challenges)) or 0
            ),
            completed_source_challenge_count=int(
                probe_field("completed_source_challenge_count", 0) or 0
            ),
            coverage_complete=bool(probe_field("coverage_complete", False)),
            coverage_failures=tuple(failures),
        )
