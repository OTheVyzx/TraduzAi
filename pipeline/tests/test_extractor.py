import zipfile
from pathlib import Path

import pytest

from pipeline.extractor.extractor import extract


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image-bytes")


def test_extract_rejects_traduzai_project_directory(tmp_path):
    source = tmp_path / "exported-project"
    _write_image(source / "originals" / "001.jpg")
    _write_image(source / "translated" / "001.jpg")
    (source / "project.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Abrir projeto"):
        extract(source, tmp_path / "work")


def test_extract_rejects_traduzai_project_archive(tmp_path):
    source = tmp_path / "traduzido.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("project.json", "{}")
        zf.writestr("originals/001.jpg", b"original")
        zf.writestr("translated/001.jpg", b"translated")

    with pytest.raises(ValueError, match="Abrir projeto"):
        extract(source, tmp_path / "work")


def test_extract_allows_plain_image_directory(tmp_path):
    source = tmp_path / "plain-source"
    _write_image(source / "001.jpg")

    image_files, tmp_dir = extract(source, tmp_path / "work")

    assert [path.name for path in image_files] == ["001.jpg"]
    assert tmp_dir.name == "_tmp"
