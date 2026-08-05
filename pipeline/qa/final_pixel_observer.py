"""Independent observation of text visible in persisted final-page bytes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

from ownership.hash_contract import canonical_page_sha256
from strip.page_surface_geometry import PageSurfaceGeometry


class TerminalVerificationIdentityError(ValueError):
    """Raised when final OCR evidence belongs to other candidate pixels."""


class TerminalVerificationInfrastructureError(RuntimeError):
    """Raised when terminal OCR lacks a fresh physical full-page attempt."""


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
    observation_space: str = "logical_page"
    page_surface_geometry_sha256: str = ""
    geometry_projection_count: int = 0
    root_input_pixel_sha256: str = ""
    request_scoped: bool = False

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
        page_surface_geometry: PageSurfaceGeometry | dict[str, Any] | None = None,
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
        page_surface_geometry: PageSurfaceGeometry | dict[str, Any] | None = None,
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
        root_input_pixel_sha256 = canonical_page_sha256(persisted_image_rgb)
        geometry = page_surface_geometry
        if isinstance(geometry, dict):
            geometry = PageSurfaceGeometry.from_dict(geometry)
        if geometry is not None and not isinstance(geometry, PageSurfaceGeometry):
            raise TypeError("final pixel observer requires PageSurfaceGeometry")
        if geometry is not None and image_rgb.shape[:2] != (
            geometry.frame_height,
            geometry.frame_width,
        ):
            raise ValueError("persisted final page shape does not match page surface geometry")

        prepared_challenges: list[dict[str, Any]] = []
        for raw in source_challenges:
            challenge = copy.deepcopy(dict(raw))
            if geometry is not None:
                raw_bbox = challenge.get("challenge_bbox_logical") or challenge.get("bbox_page")
                if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                    raise ValueError("source challenge is missing logical bbox")
                logical_bbox = tuple(int(value) for value in raw_bbox)
                frame_bbox = geometry.logical_bbox_to_frame(logical_bbox)
                challenge["coordinate_space"] = "logical_page"
                challenge["challenge_bbox_logical"] = list(logical_bbox)
                challenge["artifact_bbox_frame"] = list(frame_bbox)
                challenge["page_surface_geometry_sha256"] = geometry.geometry_sha256
            prepared_challenges.append(challenge)

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
            source_challenges=prepared_challenges,
            page_id=str(page_id or ""),
            page_number=int(page_number or 0),
            source_language=str(source_language),
            page_surface_geometry=(geometry.to_dict() if geometry is not None else None),
            request_scoped=True,
            root_input_pixel_sha256=root_input_pixel_sha256,
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
        probe_is_request_scoped = bool(probe_field("request_scoped", False))
        probe_root_sha256 = str(probe_field("root_input_pixel_sha256", "") or "")
        if probe_root_sha256 and probe_root_sha256 != root_input_pixel_sha256:
            raise TerminalVerificationIdentityError(
                "final OCR probe root does not match persisted candidate pixels"
            )
        if probe_is_request_scoped:
            qualifying_full_page = [
                attempt
                for attempt in attempts
                if str(attempt.get("variant_id") or "") == "full_page"
                and bool(attempt.get("provider_called"))
                and not bool(attempt.get("cache_hit"))
                and str(attempt.get("root_input_pixel_sha256") or "")
                == root_input_pixel_sha256
                and str(attempt.get("input_pixel_sha256") or "")
                == root_input_pixel_sha256
            ]
            if not qualifying_full_page:
                raise TerminalVerificationInfrastructureError(
                    "terminal OCR requires an uncached physical full-page attempt"
                )
        observed_blocks = list(detected_blocks)
        if geometry is not None:
            observed_blocks = []
            for raw_block in detected_blocks:
                block = copy.deepcopy(dict(raw_block))
                raw_bbox = block.get("bbox")
                try:
                    frame_bbox = tuple(int(value) for value in raw_bbox)
                    logical_bbox = geometry.frame_bbox_to_logical(frame_bbox)
                except (TypeError, ValueError):
                    continue
                block["artifact_bbox_frame"] = list(frame_bbox)
                block["bbox"] = list(logical_bbox)
                block["coordinate_space"] = "logical_page"
                observed_blocks.append(block)
        return FinalPixelObservation(
            image_path=path,
            persisted_sha256=persisted_sha256,
            image_rgb=persisted_image_rgb,
            detected_blocks=tuple(observed_blocks),
            ocr_records=tuple(raw_records),
            source_language=str(source_language),
            page_id=str(page_id or ""),
            page_number=int(page_number or 0),
            detected_block_count=len(observed_blocks),
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
            observation_space=str(probe_field("observation_space", "logical_page")),
            page_surface_geometry_sha256=(
                geometry.geometry_sha256 if geometry is not None else ""
            ),
            geometry_projection_count=int(
                probe_field("geometry_projection_count", 1 if geometry is not None else 0) or 0
            ),
            root_input_pixel_sha256=root_input_pixel_sha256,
            request_scoped=probe_is_request_scoped,
        )
