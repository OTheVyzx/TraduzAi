from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, PngImagePlugin

from ownership.chapter_contract import ChapterSourceManifest
from ownership.hash_contract import canonical_source_tree_sha256, sha256_file


def _write_png(path: Path, color: tuple[int, int, int], *, note: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (7, 5), color)
    pnginfo = None
    if note is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("note", note)
    image.save(path, pnginfo=pnginfo)


def test_chapter_source_manifest_uses_relative_paths_real_hashes_and_stable_order(tmp_path):
    root = tmp_path / "source"
    paths = (
        root / "chapter-a" / "001.png",
        root / "chapter-b" / "001.png",
        root / "010.png",
    )
    for index, path in enumerate(paths, 1):
        _write_png(path, (index * 20, index * 30, index * 40))

    manifest = ChapterSourceManifest.from_extracted_pages(
        paths,
        root,
        run_id="content-lineage-1",
        execution_id="execution-1",
    )

    assert [entry.page_id for entry in manifest.pages] == ["page_001", "page_002", "page_003"]
    assert [entry.relative_source_path for entry in manifest.pages] == [
        "chapter-a/001.png",
        "chapter-b/001.png",
        "010.png",
    ]
    assert all(not Path(entry.relative_source_path).is_absolute() for entry in manifest.pages)
    assert [entry.source_file_sha256 for entry in manifest.pages] == [sha256_file(path) for path in paths]
    assert all(re.fullmatch(r"[0-9a-f]{64}", entry.page_source_sha256) for entry in manifest.pages)
    assert manifest.source_tree_sha256 == canonical_source_tree_sha256(paths, root)
    assert manifest.source_page_count == 3
    assert manifest.sha256 == ChapterSourceManifest.from_canonical_json_bytes(
        manifest.canonical_json_bytes
    ).sha256


def test_page_identity_ignores_lossless_file_metadata_but_file_audit_hash_does_not(tmp_path):
    root = tmp_path / "source"
    first = root / "001-a.png"
    second = root / "001-b.png"
    _write_png(first, (20, 40, 60), note="encoding-a")
    _write_png(second, (20, 40, 60), note="encoding-b")

    manifest = ChapterSourceManifest.from_extracted_pages(
        (first, second), root, run_id="run-a", execution_id="exec-a"
    )

    assert manifest.pages[0].page_source_sha256 == manifest.pages[1].page_source_sha256
    assert manifest.pages[0].source_file_sha256 != manifest.pages[1].source_file_sha256


def test_source_manifest_rejects_order_or_path_tamper(tmp_path):
    root = tmp_path / "source"
    first = root / "001.png"
    second = root / "002.png"
    _write_png(first, (1, 2, 3))
    _write_png(second, (4, 5, 6))
    manifest = ChapterSourceManifest.from_extracted_pages(
        (first, second), root, run_id="run-a", execution_id="exec-a"
    )
    payload = manifest.to_canonical_dict()
    payload["pages"] = list(reversed(payload["pages"]))

    import json
    import pytest

    with pytest.raises(ValueError, match="source tree|order|page_id"):
        ChapterSourceManifest.from_canonical_json_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
