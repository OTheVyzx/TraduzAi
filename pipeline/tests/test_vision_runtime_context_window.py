from __future__ import annotations

import hashlib

from PIL import Image
import pytest


def _write_page(path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (20, 30), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _members(tmp_path):
    previous = tmp_path / "previous.png"
    target = tmp_path / "target.png"
    previous_sha = _write_page(previous, (20, 30, 40))
    target_sha = _write_page(target, (200, 210, 220))
    return [
        {
            "page_id": "page_previous",
            "source_path": str(previous),
            "source_sha256": previous_sha,
            "crop_bbox": [0, 20, 20, 30],
            "virtual_bbox": [0, -10, 20, 0],
            "role": "authenticated_context",
        },
        {
            "page_id": "page_target",
            "source_path": str(target),
            "source_sha256": target_sha,
            "crop_bbox": [0, 0, 20, 15],
            "virtual_bbox": [0, 0, 20, 15],
            "role": "target",
        },
    ]


def test_context_window_authenticates_sources_and_reverses_coordinates(tmp_path) -> None:
    from vision_runtime.context_window import AuthenticatedContextWindow

    window = AuthenticatedContextWindow.build(_members(tmp_path))

    assert window.size == (20, 25)
    assert window.map_virtual_bbox((2, -8, 10, -2)) == (
        "page_previous", (2, 22, 10, 28)
    )
    assert window.map_virtual_bbox((1, 2, 8, 9)) == (
        "page_target", (1, 2, 8, 9)
    )
    assert window.can_write("page_previous") is False
    assert window.can_write("page_target") is True


def test_window_dependency_changes_when_authenticated_neighbor_changes(tmp_path) -> None:
    from vision_runtime.context_window import AuthenticatedContextWindow

    members = _members(tmp_path)
    first = AuthenticatedContextWindow.build(members)
    members[0]["source_sha256"] = _write_page(
        tmp_path / "previous.png", (21, 30, 40)
    )
    second = AuthenticatedContextWindow.build(members)

    assert first.dependency_sha256 != second.dependency_sha256


def test_context_window_renders_only_bounded_local_crops(tmp_path) -> None:
    from vision_runtime.context_window import AuthenticatedContextWindow

    window = AuthenticatedContextWindow.build(_members(tmp_path))
    rendered = window.render_rgb()

    assert rendered.size == (20, 25)
    assert rendered.getpixel((5, 5)) == (20, 30, 40)
    assert rendered.getpixel((5, 12)) == (200, 210, 220)


def test_context_window_rejects_unaligned_or_gapped_members(tmp_path) -> None:
    from vision_runtime.context_window import AuthenticatedContextWindow

    members = _members(tmp_path)
    members[1]["virtual_bbox"] = [1, 2, 21, 17]

    with pytest.raises(ValueError, match="contiguous and width-aligned"):
        AuthenticatedContextWindow.build(members)


def test_virtual_bbox_cannot_cross_member_write_boundaries(tmp_path) -> None:
    from vision_runtime.context_window import AuthenticatedContextWindow

    window = AuthenticatedContextWindow.build(_members(tmp_path))

    with pytest.raises(ValueError, match="exactly one source member"):
        window.map_virtual_bbox((2, -2, 10, 3))
