"""Page-native chapter geometry without a chapter-sized raster allocation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np

from ownership.hash_contract import canonical_page_sha256
from strip.page_surface_geometry import PageSurfaceGeometry
from strip.types import VerticalStrip


@dataclass(frozen=True)
class ChapterPageEntry:
    """Immutable public identity plus private runtime path for one source page."""

    page_id: str
    ordinal: int
    relative_name: str
    width: int
    height: int
    pixel_sha256: str
    geometry: PageSurfaceGeometry
    virtual_y_offset: int
    _source_path: Path

    def to_public_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "ordinal": self.ordinal,
            "relative_name": self.relative_name,
            "width": self.width,
            "height": self.height,
            "pixel_sha256": self.pixel_sha256,
            "geometry": self.geometry.to_dict(),
            "virtual_y_offset": self.virtual_y_offset,
        }

    def load_framed_rgb(self) -> np.ndarray:
        bgr = cv2.imread(str(self._source_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"unable to decode source page: {self.relative_name}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if canonical_page_sha256(rgb) != self.pixel_sha256:
            raise ValueError(f"source page pixels changed after page-map creation: {self.page_id}")
        return self.geometry.logical_array_to_frame(rgb, fill_value=255)

    def load_as_local_strip(self) -> VerticalStrip:
        image = self.load_framed_rgb()
        return VerticalStrip(
            image=image,
            width=int(self.geometry.frame_width),
            height=int(self.geometry.frame_height),
            source_page_breaks=[0, int(self.geometry.frame_height)],
            page_x_offsets=[int(self.geometry.content_origin_xy[0])],
            source_page_widths=[int(self.width)],
            page_number_offset=int(self.ordinal) - 1,
            raster_mode="page_map_v1",
        )


@dataclass(frozen=True)
class ChapterPageMap:
    """Ordered page catalog whose virtual Y axis never owns image pixels."""

    pages: tuple[ChapterPageEntry, ...]
    frame_width: int
    virtual_height: int

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "ChapterPageMap":
        if not paths:
            raise ValueError("chapter page map requires at least one page")
        resolved = [Path(path).resolve() for path in paths]
        common_root = Path(os.path.commonpath([str(path) for path in resolved]))
        if common_root.is_file():
            common_root = common_root.parent
        metadata: list[tuple[Path, int, int, str]] = []
        for path in resolved:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"unable to decode source page: {path.name}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = (int(value) for value in image.shape[:2])
            metadata.append((path, width, height, canonical_page_sha256(image)))
            del image, bgr
        frame_width = max(width for _, width, _, _ in metadata)
        cursor = 0
        entries: list[ChapterPageEntry] = []
        for ordinal, (path, width, height, pixel_sha256) in enumerate(metadata, start=1):
            x_offset = (frame_width - width) // 2
            geometry = PageSurfaceGeometry.build(
                logical_width=width,
                logical_height=height,
                frame_width=frame_width,
                frame_height=height,
                content_origin_xy=(x_offset, 0),
            )
            try:
                relative_name = path.relative_to(common_root).as_posix()
            except ValueError:
                relative_name = path.name
            entries.append(
                ChapterPageEntry(
                    page_id=f"page_{ordinal:03d}",
                    ordinal=ordinal,
                    relative_name=relative_name,
                    width=width,
                    height=height,
                    pixel_sha256=pixel_sha256,
                    geometry=geometry,
                    virtual_y_offset=cursor,
                    _source_path=path,
                )
            )
            cursor += height
        return cls(tuple(entries), frame_width=frame_width, virtual_height=cursor)

    def to_public_dict(self) -> dict:
        return {
            "schema_version": 1,
            "chapter_raster_mode": "page_map_v1",
            "frame_width": self.frame_width,
            "virtual_height": self.virtual_height,
            "pages": [page.to_public_dict() for page in self.pages],
        }
