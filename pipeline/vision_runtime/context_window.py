"""Authenticated, reversible local context windows across source members."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from PIL import Image


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bbox(value: Any, field: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        result = ()
    if len(result) != 4 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"{field} must be a non-empty bbox")
    return result


@dataclass(frozen=True)
class ContextMember:
    page_id: str
    source_path: Path
    source_sha256: str
    crop_bbox: tuple[int, int, int, int]
    virtual_bbox: tuple[int, int, int, int]
    role: str

    @property
    def writable(self) -> bool:
        return self.role == "target"


@dataclass(frozen=True)
class AuthenticatedContextWindow:
    members: tuple[ContextMember, ...]
    dependency_sha256: str
    size: tuple[int, int]
    origin_y: int

    @classmethod
    def build(
        cls, raw_members: Sequence[Mapping[str, Any]]
    ) -> "AuthenticatedContextWindow":
        members: list[ContextMember] = []
        for raw in raw_members:
            source_path = Path(str(raw.get("source_path") or ""))
            expected_sha256 = str(raw.get("source_sha256") or "")
            if not _SHA256.fullmatch(expected_sha256):
                raise ValueError("source_sha256 must be a lowercase SHA-256")
            try:
                actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f"source cannot be authenticated: {source_path}") from exc
            if actual_sha256 != expected_sha256:
                raise ValueError(f"source hash mismatch: {source_path}")
            crop_bbox = _bbox(raw.get("crop_bbox"), "crop_bbox")
            virtual_bbox = _bbox(raw.get("virtual_bbox"), "virtual_bbox")
            if (
                crop_bbox[2] - crop_bbox[0] != virtual_bbox[2] - virtual_bbox[0]
                or crop_bbox[3] - crop_bbox[1] != virtual_bbox[3] - virtual_bbox[1]
            ):
                raise ValueError("crop and virtual bbox dimensions must match")
            with Image.open(source_path) as image:
                if not (
                    0 <= crop_bbox[0] < crop_bbox[2] <= image.width
                    and 0 <= crop_bbox[1] < crop_bbox[3] <= image.height
                ):
                    raise ValueError("crop bbox exceeds authenticated source")
            role = str(raw.get("role") or "")
            if role not in {"target", "authenticated_context"}:
                raise ValueError("context member role is invalid")
            members.append(ContextMember(
                page_id=str(raw.get("page_id") or ""),
                source_path=source_path,
                source_sha256=expected_sha256,
                crop_bbox=crop_bbox,
                virtual_bbox=virtual_bbox,
                role=role,
            ))
        members.sort(key=lambda item: (item.virtual_bbox[1], item.virtual_bbox[0]))
        if not members or sum(item.writable for item in members) != 1:
            raise ValueError("context window requires exactly one target")
        expected_x = members[0].virtual_bbox[0]
        expected_width = members[0].virtual_bbox[2] - expected_x
        previous_bottom = members[0].virtual_bbox[1]
        for member in members:
            left, top, right, bottom = member.virtual_bbox
            if (
                left != expected_x
                or right - left != expected_width
                or top != previous_bottom
            ):
                raise ValueError("context members must be contiguous and width-aligned")
            previous_bottom = bottom
        origin_y = members[0].virtual_bbox[1]
        dependency = [
            {
                "page_id": member.page_id,
                "source_sha256": member.source_sha256,
                "crop_bbox": list(member.crop_bbox),
                "virtual_bbox": list(member.virtual_bbox),
                "role": member.role,
            }
            for member in members
        ]
        return cls(
            members=tuple(members),
            dependency_sha256=_canonical_sha256(dependency),
            size=(expected_width, previous_bottom - origin_y),
            origin_y=origin_y,
        )

    def can_write(self, page_id: str) -> bool:
        return any(member.page_id == page_id and member.writable for member in self.members)

    def map_virtual_bbox(
        self, bbox: tuple[int, int, int, int]
    ) -> tuple[str, tuple[int, int, int, int]]:
        left, top, right, bottom = _bbox(bbox, "virtual bbox")
        matches = [
            member for member in self.members
            if (
                member.virtual_bbox[0] <= left < right <= member.virtual_bbox[2]
                and member.virtual_bbox[1] <= top < bottom <= member.virtual_bbox[3]
            )
        ]
        if len(matches) != 1:
            raise ValueError("virtual bbox must belong to exactly one source member")
        member = matches[0]
        virtual_left, virtual_top, _, _ = member.virtual_bbox
        crop_left, crop_top, _, _ = member.crop_bbox
        return member.page_id, (
            crop_left + left - virtual_left,
            crop_top + top - virtual_top,
            crop_left + right - virtual_left,
            crop_top + bottom - virtual_top,
        )

    def render_rgb(self) -> Image.Image:
        window = Image.new("RGB", self.size)
        for member in self.members:
            with Image.open(member.source_path) as image:
                crop = image.convert("RGB").crop(member.crop_bbox)
            x = member.virtual_bbox[0] - self.members[0].virtual_bbox[0]
            y = member.virtual_bbox[1] - self.origin_y
            window.paste(crop, (x, y))
        return window
