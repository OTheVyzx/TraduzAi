from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from integration_v1.context_consumer import consume_authenticated_context


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _handoff(first: bytes, second: bytes) -> dict:
    return {
        "schema": "traduzai.vision.authenticated-context-handoff.v1",
        "status": "complete",
        "case": "036/037",
        "window": {
            "artifact_ref": "window.png",
            "sha256": _sha(b"window"),
            "width": 690,
            "height": 660,
            "virtual_origin": [0, -240],
        },
        "members": [
            {
                "page_id": "page_036",
                "source_member": "036.webp",
                "source_sha256": _sha(first),
                "crop_bbox": [0, 1770, 690, 2010],
                "virtual_bbox": [0, -240, 690, 0],
                "role": "authenticated_context",
                "write_authority": False,
            },
            {
                "page_id": "page_037",
                "source_member": "037.webp",
                "source_sha256": _sha(second),
                "crop_bbox": [0, 0, 690, 420],
                "virtual_bbox": [0, 0, 690, 420],
                "role": "target",
                "write_authority": True,
            },
        ],
        "reversible_mapping_example": {
            "virtual_bbox": [104, 21, 372, 255],
            "page_id": "page_037",
            "source_bbox": [104, 21, 372, 255],
        },
    }


def test_consumer_verifies_both_members_and_keeps_neighbor_read_only(tmp_path):
    first, second = b"first-page", b"second-page"
    archive = tmp_path / "chapter.cbz"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("036.webp", first)
        output.writestr("037.webp", second)
    handoff = _handoff(first, second)
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    (tmp_path / "window.png").write_bytes(b"window")

    receipt = consume_authenticated_context(
        handoff_path,
        archive,
        expected_handoff_sha256=_sha(handoff_path.read_bytes()),
    )

    assert receipt["status"] == "verified"
    assert [member["source_sha256"] for member in receipt["members"]] == [
        _sha(first),
        _sha(second),
    ]
    assert receipt["write_authority_page_ids"] == ["page_037"]
    assert receipt["read_only_context_page_ids"] == ["page_036"]
    assert all(member["reversible"] for member in receipt["members"])
    assert receipt["published_pixels"] is False


def test_consumer_rejects_context_write_authority(tmp_path):
    first, second = b"first-page", b"second-page"
    archive = tmp_path / "chapter.cbz"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("036.webp", first)
        output.writestr("037.webp", second)
    handoff = _handoff(first, second)
    handoff["members"][0]["write_authority"] = True
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    (tmp_path / "window.png").write_bytes(b"window")

    with pytest.raises(ValueError, match="context member cannot have write authority"):
        consume_authenticated_context(
            handoff_path,
            archive,
            expected_handoff_sha256=_sha(handoff_path.read_bytes()),
        )
