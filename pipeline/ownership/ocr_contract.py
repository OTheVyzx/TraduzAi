"""Request-scoped immutable OCR evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping
import re
import unicodedata

import numpy as np
from numpy.typing import NDArray

from .hash_contract import (
    JSONValue,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    sha256_text,
)


BBox = tuple[int, int, int, int]
Point = tuple[int, int]


class OCRRequestIdentityError(ValueError):
    """Raised when evidence does not belong to its declared OCR request."""


class OCRInputPixelIdentityError(ValueError):
    """Raised before inference when declared and physical pixels differ."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_ocr_payload_text(text: str) -> str:
    """Return the stable text representation bound by payload_sha256."""

    return " ".join(unicodedata.normalize("NFKC", str(text or "")).split())


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"extras contain non-JSON value: {type(value).__name__}")


def _thaw_json(value) -> JSONValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_identity_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OCRRequestIdentityError(f"{field_name} must be non-empty")


def _require_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.fullmatch(str(value or "")):
        raise OCRRequestIdentityError(f"{field_name} must be a lowercase SHA-256")


def _as_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    import cv2

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("OCR transform pixels must use uint8 samples")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("OCR transform pixels must be HxW, HxWx1, HxWx3 or HxWx4")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("OCR transform pixels cannot be empty")
    return np.ascontiguousarray(array, dtype=np.uint8)


@dataclass(frozen=True)
class OCRRequest:
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    invocation_id: str
    provider_family: str

    def __post_init__(self) -> None:
        for name in ("run_id", "origin_execution_id", "page_id", "invocation_id", "provider_family"):
            _require_identity_text(getattr(self, name), name)
        _require_sha256(self.page_source_sha256, "page_source_sha256")
        _require_sha256(self.root_input_pixel_sha256, "root_input_pixel_sha256")

    @property
    def identity(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.root_input_pixel_sha256,
            self.invocation_id,
            self.provider_family,
        )


@dataclass(frozen=True)
class OCRTransformOperation:
    kind: Literal[
        "identity", "crop", "resize", "grayscale_to_rgb", "invert",
        "rotate_affine", "deskew_affine", "mask_rectangles",
    ]
    bbox_page: BBox | None = None
    output_size: tuple[int, int] | None = None
    interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos4"] | None = None
    border_mode: Literal["constant", "replicate", "reflect", "reflect101"] | None = None
    border_value_rgb: tuple[int, int, int] | None = None
    affine_matrix_fixed_1e6: tuple[int, int, int, int, int, int] | None = None
    algorithm_id: str = ""
    mask_bboxes: tuple[BBox, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "crop":
            if self.bbox_page is None or len(self.bbox_page) != 4:
                raise ValueError("crop transform requires bbox_page")
        elif self.kind == "resize":
            if self.output_size is None or self.interpolation is None:
                raise ValueError("resize transform requires output_size and interpolation")
        elif self.kind in {"rotate_affine", "deskew_affine"}:
            required = (
                self.output_size,
                self.interpolation,
                self.border_mode,
                self.border_value_rgb,
                self.affine_matrix_fixed_1e6,
                self.algorithm_id,
            )
            if any(value is None or value == "" for value in required):
                raise ValueError(f"{self.kind} transform is incomplete")
            if len(self.affine_matrix_fixed_1e6 or ()) != 6:
                raise ValueError("affine transform requires six fixed-point coefficients")
        elif self.kind == "mask_rectangles":
            if not self.mask_bboxes or any(len(box) != 4 or box[2] <= box[0] or box[3] <= box[1] for box in self.mask_bboxes):
                raise ValueError("mask_rectangles requires valid mask_bboxes")
        elif self.kind not in {"identity", "grayscale_to_rgb", "invert"}:
            raise ValueError(f"unsupported OCR transform operation: {self.kind}")
        if self.output_size is not None and (
            len(self.output_size) != 2 or self.output_size[0] <= 0 or self.output_size[1] <= 0
        ):
            raise ValueError("transform output_size must contain positive width and height")

    def to_json(self) -> dict[str, JSONValue]:
        payload = {
            "affine_matrix_fixed_1e6": list(self.affine_matrix_fixed_1e6) if self.affine_matrix_fixed_1e6 else None,
            "algorithm_id": self.algorithm_id,
            "bbox_page": list(self.bbox_page) if self.bbox_page else None,
            "border_mode": self.border_mode,
            "border_value_rgb": list(self.border_value_rgb) if self.border_value_rgb else None,
            "interpolation": self.interpolation,
            "kind": self.kind,
            "output_size": list(self.output_size) if self.output_size else None,
        }
        if self.mask_bboxes:
            payload["mask_bboxes"] = [list(box) for box in self.mask_bboxes]
        return payload


@dataclass(frozen=True)
class OCRTransformSpec:
    schema_version: int
    operations: tuple[OCRTransformOperation, ...]
    canonical_json_bytes: bytes
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported OCR transform schema version")
        if not self.operations:
            raise ValueError("OCR transform spec requires at least one operation")
        expected_bytes = canonical_json_bytes(
            {"operations": [operation.to_json() for operation in self.operations], "schema_version": 1}
        )
        if self.canonical_json_bytes != expected_bytes:
            raise ValueError("OCR transform canonical bytes mismatch")
        if self.sha256 != sha256_bytes(expected_bytes):
            raise ValueError("OCR transform SHA-256 mismatch")

    @classmethod
    def build(cls, operations: tuple[OCRTransformOperation, ...]) -> "OCRTransformSpec":
        normalized = tuple(operations)
        payload: JSONValue = {
            "operations": [operation.to_json() for operation in normalized],
            "schema_version": 1,
        }
        encoded = canonical_json_bytes(payload)
        return cls(1, normalized, encoded, sha256_bytes(encoded))

    def replay(self, root_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        import cv2

        interpolation_codes = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        border_codes = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect101": cv2.BORDER_REFLECT_101,
        }
        current = _as_rgb(root_rgb)
        for operation in self.operations:
            if operation.kind == "identity":
                current = current.copy()
            elif operation.kind == "crop":
                x1, y1, x2, y2 = operation.bbox_page or (0, 0, 0, 0)
                if x1 < 0 or y1 < 0 or x2 > current.shape[1] or y2 > current.shape[0] or x2 <= x1 or y2 <= y1:
                    raise ValueError("crop transform bbox is outside its parent pixels")
                current = current[y1:y2, x1:x2].copy()
            elif operation.kind == "resize":
                current = cv2.resize(
                    current,
                    operation.output_size,
                    interpolation=interpolation_codes[operation.interpolation],
                )
            elif operation.kind == "grayscale_to_rgb":
                gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                current = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            elif operation.kind == "invert":
                current = cv2.bitwise_not(current)
            elif operation.kind == "mask_rectangles":
                current = current.copy()
                for x1, y1, x2, y2 in operation.mask_bboxes:
                    if x1 < 0 or y1 < 0 or x2 > current.shape[1] or y2 > current.shape[0]:
                        raise ValueError("mask rectangle is outside its parent pixels")
                    current[y1:y2, x1:x2] = 255
            else:
                matrix = np.asarray(operation.affine_matrix_fixed_1e6, dtype=np.float64).reshape(2, 3) / 1_000_000.0
                current = cv2.warpAffine(
                    current,
                    matrix,
                    operation.output_size,
                    flags=interpolation_codes[operation.interpolation],
                    borderMode=border_codes[operation.border_mode],
                    borderValue=operation.border_value_rgb,
                )
            current = _as_rgb(current)
        return current


@dataclass(frozen=True)
class OCRAttempt:
    attempt_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    invocation_id: str
    provider_family: str
    variant_id: str
    input_pixel_sha256: str
    parent_input_pixel_sha256: str
    input_bbox_page: BBox | None
    input_kind: str
    transform_spec: OCRTransformSpec
    input_width: int
    input_height: int
    input_mode: Literal["RGB"]
    provider_called: bool
    cache_hit: bool

    def __post_init__(self) -> None:
        for name in (
            "attempt_id", "run_id", "origin_execution_id", "page_id", "invocation_id",
            "provider_family", "variant_id", "input_kind",
        ):
            _require_identity_text(getattr(self, name), name)
        for name in (
            "page_source_sha256", "root_input_pixel_sha256", "input_pixel_sha256",
            "parent_input_pixel_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.input_width <= 0 or self.input_height <= 0 or self.input_mode != "RGB":
            raise OCRRequestIdentityError("OCR attempt input geometry is invalid")
        if self.provider_called and self.cache_hit:
            raise OCRRequestIdentityError("a physical provider call cannot be a cache hit")

    @property
    def identity(self) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.root_input_pixel_sha256,
            self.invocation_id,
            self.provider_family,
            self.attempt_id,
            self.variant_id,
            self.input_pixel_sha256,
        )

    @property
    def qualifies_as_fresh_physical_inference(self) -> bool:
        return self.provider_called and not self.cache_hit

    @property
    def request_identity(self) -> tuple[str, str, str, str, str, str, str]:
        return self.identity[:7]

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "attempt_id": self.attempt_id,
            "cache_hit": self.cache_hit,
            "input_bbox_page": list(self.input_bbox_page) if self.input_bbox_page else None,
            "input_height": self.input_height,
            "input_kind": self.input_kind,
            "input_mode": self.input_mode,
            "input_pixel_sha256": self.input_pixel_sha256,
            "input_width": self.input_width,
            "invocation_id": self.invocation_id,
            "origin_execution_id": self.origin_execution_id,
            "page_id": self.page_id,
            "page_source_sha256": self.page_source_sha256,
            "parent_input_pixel_sha256": self.parent_input_pixel_sha256,
            "provider_called": self.provider_called,
            "provider_family": self.provider_family,
            "root_input_pixel_sha256": self.root_input_pixel_sha256,
            "run_id": self.run_id,
            "transform_spec_canonical_json": self.transform_spec.canonical_json_bytes.decode("utf-8"),
            "transform_spec_sha256": self.transform_spec.sha256,
            "variant_id": self.variant_id,
        }


@dataclass(frozen=True)
class OCRObservationRecord:
    observation_id: str
    attempt_id: str
    run_id: str
    origin_execution_id: str
    page_id: str
    page_source_sha256: str
    root_input_pixel_sha256: str
    input_pixel_sha256: str
    invocation_id: str
    provider_family: str
    variant_id: str
    payload_sha256: str
    text: str
    confidence: float
    bbox_page: BBox
    polygon_page: tuple[Point, ...]
    source: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id", "attempt_id", "run_id", "origin_execution_id", "page_id",
            "invocation_id", "provider_family", "variant_id", "source",
        ):
            _require_identity_text(getattr(self, name), name)
        for name in (
            "page_source_sha256", "root_input_pixel_sha256", "input_pixel_sha256", "payload_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise OCRRequestIdentityError("OCR record confidence must be between zero and one")

    @property
    def request_identity(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.run_id,
            self.origin_execution_id,
            self.page_id,
            self.page_source_sha256,
            self.root_input_pixel_sha256,
            self.invocation_id,
            self.provider_family,
        )

    @property
    def attempt_identity(self) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        return self.request_identity + (self.attempt_id, self.variant_id, self.input_pixel_sha256)


@dataclass(frozen=True)
class OCRBlock:
    block_id: str
    text: str
    confidence: float
    bbox_page: BBox
    polygon_page: tuple[Point, ...] = ()
    extras: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identity_text(self.block_id, "block_id")
        object.__setattr__(self, "extras", _freeze_json(self.extras))


@dataclass(frozen=True)
class OCRDiagnostics:
    provider: str
    extras: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identity_text(self.provider, "diagnostics.provider")
        object.__setattr__(self, "extras", _freeze_json(self.extras))


@dataclass(frozen=True)
class OCRInvocationResult:
    request: OCRRequest
    blocks: tuple[OCRBlock, ...]
    observations: tuple[OCRObservationRecord, ...]
    full_page_lines: tuple[OCRObservationRecord, ...]
    attempts: tuple[OCRAttempt, ...]
    attempt_chain_sha256: str
    diagnostics: OCRDiagnostics

    @classmethod
    def build(
        cls,
        *,
        request: OCRRequest,
        blocks: tuple[OCRBlock, ...] = (),
        observations: tuple[OCRObservationRecord, ...] = (),
        full_page_lines: tuple[OCRObservationRecord, ...] = (),
        attempts: tuple[OCRAttempt, ...] = (),
        diagnostics: OCRDiagnostics | None = None,
    ) -> "OCRInvocationResult":
        normalized_blocks = tuple(blocks)
        normalized_observations = tuple(observations)
        normalized_lines = tuple(full_page_lines)
        normalized_attempts = tuple(attempts)
        attempts_by_id: dict[str, OCRAttempt] = {}
        for attempt in normalized_attempts:
            if attempt.attempt_id in attempts_by_id:
                raise OCRRequestIdentityError(f"duplicate OCR attempt_id: {attempt.attempt_id}")
            if attempt.request_identity != request.identity:
                raise OCRRequestIdentityError("OCR attempt belongs to a different logical request")
            attempts_by_id[attempt.attempt_id] = attempt
        observation_ids: set[str] = set()
        for record in normalized_observations:
            if record.observation_id in observation_ids:
                raise OCRRequestIdentityError(f"duplicate OCR observation_id: {record.observation_id}")
            observation_ids.add(record.observation_id)
        for record in (*normalized_observations, *normalized_lines):
            if record.request_identity != request.identity:
                raise OCRRequestIdentityError("OCR record belongs to a different logical request")
            attempt = attempts_by_id.get(record.attempt_id)
            if attempt is None:
                raise OCRRequestIdentityError("OCR record references an unknown physical attempt")
            if record.attempt_identity != attempt.identity:
                raise OCRRequestIdentityError("OCR record physical identity does not match its attempt")
            expected_payload = sha256_text(normalize_ocr_payload_text(record.text))
            if record.payload_sha256 != expected_payload:
                raise OCRRequestIdentityError("OCR record payload SHA-256 mismatch")
        chain: JSONValue = [attempt.to_json() for attempt in normalized_attempts]
        return cls(
            request=request,
            blocks=normalized_blocks,
            observations=normalized_observations,
            full_page_lines=normalized_lines,
            attempts=normalized_attempts,
            attempt_chain_sha256=canonical_json_sha256(chain),
            diagnostics=diagnostics or OCRDiagnostics(request.provider_family),
        )

    @property
    def run_id(self) -> str:
        return self.request.run_id

    @property
    def origin_execution_id(self) -> str:
        return self.request.origin_execution_id

    @property
    def page_id(self) -> str:
        return self.request.page_id

    @property
    def page_source_sha256(self) -> str:
        return self.request.page_source_sha256

    @property
    def root_input_pixel_sha256(self) -> str:
        return self.request.root_input_pixel_sha256

    @property
    def invocation_id(self) -> str:
        return self.request.invocation_id

    @property
    def provider_family(self) -> str:
        return self.request.provider_family
