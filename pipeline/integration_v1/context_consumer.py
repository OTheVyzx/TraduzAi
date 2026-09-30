"""Consume an authenticated cross-page Vision handoff without write escalation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import zipfile

from ownership.hash_contract import canonical_json_sha256


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return candidate


def _bbox(value: Any, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain four integers")
    x1, y1, x2, y2 = value
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{field} is empty")
    return x1, y1, x2, y2


def _mapping(member: Mapping[str, Any]) -> dict[str, Any]:
    source = _bbox(member.get("crop_bbox"), "crop_bbox")
    virtual = _bbox(member.get("virtual_bbox"), "virtual_bbox")
    if source[2] - source[0] != virtual[2] - virtual[0] or source[3] - source[1] != virtual[3] - virtual[1]:
        raise ValueError("context transform must preserve member dimensions")
    offset = (virtual[0] - source[0], virtual[1] - source[1])
    mapped = tuple(
        coordinate + offset[index % 2]
        for index, coordinate in enumerate(source)
    )
    restored = tuple(
        coordinate - offset[index % 2]
        for index, coordinate in enumerate(mapped)
    )
    if mapped != virtual or restored != source:
        raise ValueError("context transform is not reversible")
    return {
        "source_bbox": list(source),
        "virtual_bbox": list(virtual),
        "offset_xy": list(offset),
        "reversible": True,
    }


def consume_authenticated_context(
    handoff_path: str | Path,
    source_cbz: str | Path,
    *,
    expected_handoff_sha256: str,
) -> dict[str, Any]:
    """Verify member bytes, reversible coordinates and page-scoped write authority."""

    handoff_file = Path(handoff_path)
    archive_path = Path(source_cbz)
    expected_handoff_sha256 = _require_sha256(expected_handoff_sha256, "handoff_sha256")
    actual_handoff_sha256 = _file_sha256(handoff_file)
    if actual_handoff_sha256 != expected_handoff_sha256:
        raise ValueError("handoff SHA-256 mismatch")
    handoff = json.loads(handoff_file.read_text(encoding="utf-8"))
    if handoff.get("schema") != "traduzai.vision.authenticated-context-handoff.v1":
        raise ValueError("unsupported authenticated context handoff")
    if handoff.get("status") != "complete":
        raise ValueError("authenticated context handoff is not complete")

    window = handoff.get("window") or {}
    artifact_ref = str(window.get("artifact_ref") or "")
    if not artifact_ref or Path(artifact_ref).name != artifact_ref:
        raise ValueError("window artifact_ref must be a local filename")
    window_path = handoff_file.parent / artifact_ref
    expected_window_sha256 = _require_sha256(window.get("sha256"), "window.sha256")
    if _file_sha256(window_path) != expected_window_sha256:
        raise ValueError("window artifact SHA-256 mismatch")

    members = handoff.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("authenticated context requires target and context members")
    verified_members: list[dict[str, Any]] = []
    write_authority_page_ids: list[str] = []
    read_only_context_page_ids: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in members:
            page_id = str(member.get("page_id") or "")
            source_member = str(member.get("source_member") or "")
            role = str(member.get("role") or "")
            write_authority = member.get("write_authority") is True
            if not page_id or not source_member:
                raise ValueError("context member identity is incomplete")
            if role == "authenticated_context" and write_authority:
                raise ValueError("context member cannot have write authority")
            if role == "target" and not write_authority:
                raise ValueError("target member requires write authority")
            source_bytes = archive.read(source_member)
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            if source_sha256 != _require_sha256(member.get("source_sha256"), "member.source_sha256"):
                raise ValueError(f"source member SHA-256 mismatch: {source_member}")
            mapping = _mapping(member)
            verified_members.append({
                "page_id": page_id,
                "source_member": source_member,
                "source_sha256": source_sha256,
                "role": role,
                "write_authority": write_authority,
                **mapping,
            })
            if write_authority:
                write_authority_page_ids.append(page_id)
            else:
                read_only_context_page_ids.append(page_id)

    if len(write_authority_page_ids) != 1:
        raise ValueError("authenticated context requires exactly one writable target")
    payload = {
        "schema": "traduzai.integration.authenticated-context-receipt.v1",
        "status": "verified",
        "handoff_sha256": actual_handoff_sha256,
        "source_cbz_sha256": _file_sha256(archive_path),
        "window_sha256": expected_window_sha256,
        "members": verified_members,
        "write_authority_page_ids": write_authority_page_ids,
        "read_only_context_page_ids": read_only_context_page_ids,
        "published_pixels": False,
        "human_approval_claimed": False,
    }
    return {**payload, "receipt_sha256": canonical_json_sha256(payload)}


def write_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(receipt), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff")
    parser.add_argument("source_cbz")
    parser.add_argument("output")
    parser.add_argument("--handoff-sha256", required=True)
    args = parser.parse_args()
    receipt = consume_authenticated_context(
        args.handoff,
        args.source_cbz,
        expected_handoff_sha256=args.handoff_sha256,
    )
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
