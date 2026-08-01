"""Lossless transforms between logical page pixels and published page frames."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


BBox = tuple[int, int, int, int]
Point = tuple[int | float, int | float]


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _origin(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("content_origin_xy must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("content_origin_xy must contain two integers")
    x, y = int(value[0]), int(value[1])
    if x < 0 or y < 0:
        raise ValueError("content_origin_xy must be non-negative")
    return x, y


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _bbox(value: Any, *, width: int, height: int, field: str) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} bbox must contain four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} bbox must contain four integers")
    x1, y1, x2, y2 = (int(item) for item in value)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError(f"{field} bbox is outside its coordinate space")
    return x1, y1, x2, y2


def _polygon(
    value: Sequence[Sequence[int | float]],
    *,
    width: int,
    height: int,
    field: str,
) -> tuple[Point, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} polygon must contain points")
    points: list[Point] = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"{field} polygon point is malformed")
        x, y = raw
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or not 0 <= float(x) <= width
            or not 0 <= float(y) <= height
        ):
            raise ValueError(f"{field} polygon point is outside its coordinate space")
        points.append((x, y))
    return tuple(points)


@dataclass(frozen=True)
class PageSurfaceGeometry:
    schema_version: int
    logical_space: str
    artifact_space: str
    logical_width: int
    logical_height: int
    frame_width: int
    frame_height: int
    content_origin_xy: tuple[int, int]
    content_bbox_frame: BBox
    geometry_sha256: str

    @classmethod
    def build(
        cls,
        *,
        logical_width: int,
        logical_height: int,
        frame_width: int,
        frame_height: int,
        content_origin_xy: tuple[int, int],
    ) -> "PageSurfaceGeometry":
        logical_width = _positive_int(logical_width, field="logical_width")
        logical_height = _positive_int(logical_height, field="logical_height")
        frame_width = _positive_int(frame_width, field="frame_width")
        frame_height = _positive_int(frame_height, field="frame_height")
        origin = _origin(content_origin_xy)
        content_bbox = (
            origin[0],
            origin[1],
            origin[0] + logical_width,
            origin[1] + logical_height,
        )
        if content_bbox[2] > frame_width or content_bbox[3] > frame_height:
            raise ValueError("logical content is outside frame")
        contract = {
            "schema_version": 1,
            "logical_space": "logical_page",
            "artifact_space": "framed_page",
            "logical_width": logical_width,
            "logical_height": logical_height,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "content_origin_xy": list(origin),
            "content_bbox_frame": list(content_bbox),
        }
        return cls(
            schema_version=1,
            logical_space="logical_page",
            artifact_space="framed_page",
            logical_width=logical_width,
            logical_height=logical_height,
            frame_width=frame_width,
            frame_height=frame_height,
            content_origin_xy=origin,
            content_bbox_frame=content_bbox,
            geometry_sha256=_canonical_sha256(contract),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PageSurfaceGeometry":
        required = {
            "schema_version", "logical_space", "artifact_space",
            "logical_width", "logical_height", "frame_width", "frame_height",
            "content_origin_xy", "content_bbox_frame", "geometry_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("page surface geometry schema is incomplete")
        if value.get("schema_version") != 1:
            raise ValueError("page surface geometry schema version is unsupported")
        if value.get("logical_space") != "logical_page" or value.get("artifact_space") != "framed_page":
            raise ValueError("page surface geometry coordinate spaces are invalid")
        rebuilt = cls.build(
            logical_width=value["logical_width"],
            logical_height=value["logical_height"],
            frame_width=value["frame_width"],
            frame_height=value["frame_height"],
            content_origin_xy=tuple(value["content_origin_xy"]),
        )
        serialized_bbox = tuple(value["content_bbox_frame"])
        if serialized_bbox != rebuilt.content_bbox_frame:
            raise ValueError("content_bbox_frame does not match origin and logical size")
        if str(value.get("geometry_sha256") or "") != rebuilt.geometry_sha256:
            raise ValueError("page surface geometry hash mismatch")
        return rebuilt

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_space": self.logical_space,
            "artifact_space": self.artifact_space,
            "logical_width": self.logical_width,
            "logical_height": self.logical_height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "content_origin_xy": list(self.content_origin_xy),
            "content_bbox_frame": list(self.content_bbox_frame),
            "geometry_sha256": self.geometry_sha256,
        }

    def logical_bbox_to_frame(self, bbox: BBox) -> BBox:
        x1, y1, x2, y2 = _bbox(
            bbox,
            width=self.logical_width,
            height=self.logical_height,
            field="logical",
        )
        ox, oy = self.content_origin_xy
        return x1 + ox, y1 + oy, x2 + ox, y2 + oy

    def frame_bbox_to_logical(self, bbox: BBox) -> BBox:
        x1, y1, x2, y2 = _bbox(
            bbox,
            width=self.frame_width,
            height=self.frame_height,
            field="frame",
        )
        ox, oy = self.content_origin_xy
        return _bbox(
            (x1 - ox, y1 - oy, x2 - ox, y2 - oy),
            width=self.logical_width,
            height=self.logical_height,
            field="logical",
        )

    def logical_polygon_to_frame(
        self,
        polygon: Sequence[Sequence[int | float]],
    ) -> tuple[Point, ...]:
        points = _polygon(
            polygon,
            width=self.logical_width,
            height=self.logical_height,
            field="logical",
        )
        ox, oy = self.content_origin_xy
        return tuple((x + ox, y + oy) for x, y in points)

    def frame_polygon_to_logical(
        self,
        polygon: Sequence[Sequence[int | float]],
    ) -> tuple[Point, ...]:
        points = _polygon(
            polygon,
            width=self.frame_width,
            height=self.frame_height,
            field="frame",
        )
        ox, oy = self.content_origin_xy
        return _polygon(
            tuple((x - ox, y - oy) for x, y in points),
            width=self.logical_width,
            height=self.logical_height,
            field="logical",
        )

    def logical_array_to_frame(
        self,
        source: np.ndarray,
        *,
        fill_value: Any = 0,
    ) -> np.ndarray:
        array = np.asarray(source)
        if array.ndim < 2 or array.shape[:2] != (self.logical_height, self.logical_width):
            raise ValueError("logical array shape does not match page geometry")
        output = np.full(
            (self.frame_height, self.frame_width, *array.shape[2:]),
            fill_value,
            dtype=array.dtype,
        )
        ox, oy = self.content_origin_xy
        output[oy:oy + self.logical_height, ox:ox + self.logical_width] = array
        return output

    def frame_array_to_logical(self, source: np.ndarray) -> np.ndarray:
        array = np.asarray(source)
        if array.ndim < 2 or array.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError("frame array shape does not match page geometry")
        ox, oy = self.content_origin_xy
        return array[
            oy:oy + self.logical_height,
            ox:ox + self.logical_width,
            ...,
        ].copy()
