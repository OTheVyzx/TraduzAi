import zipfile
from pathlib import Path

import pytest
from PIL import Image

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


def _write_valid_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color).save(path)


@pytest.mark.parametrize("input_kind", ["directory", "archive"])
def test_extractor_preserves_nested_duplicate_basenames_without_overwrite(tmp_path, input_kind):
    source_tree = tmp_path / "source-tree"
    _write_valid_image(source_tree / "chapter-a" / "001.png", (255, 0, 0))
    _write_valid_image(source_tree / "chapter-b" / "001.png", (0, 0, 255))
    if input_kind == "directory":
        source = source_tree
    else:
        source = tmp_path / "chapter.cbz"
        with zipfile.ZipFile(source, "w") as zf:
            zf.write(source_tree / "chapter-a" / "001.png", "chapter-a/001.png")
            zf.write(source_tree / "chapter-b" / "001.png", "chapter-b/001.png")

    image_files, extraction_root = extract(source, tmp_path / "work")

    assert [path.relative_to(extraction_root).as_posix() for path in image_files] == [
        "chapter-a/001.png",
        "chapter-b/001.png",
    ]
    assert image_files[0].read_bytes() != image_files[1].read_bytes()
