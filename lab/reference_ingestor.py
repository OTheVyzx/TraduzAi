from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
import zipfile


CHAPTER_RE = re.compile(r"(?i)(?:chapter|cap[ií]tulo)\s*(\d+)")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class ChapterPair:
    chapter_number: int
    source_path: str
    reference_path: str
    source_pages: int
    reference_pages: int
    reference_group: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_chapter_number(file_name: str) -> int | None:
    match = CHAPTER_RE.search(file_name)
    if not match:
        return None
    return int(match.group(1))


def extract_reference_group(file_name: str) -> str:
    stem = Path(file_name).stem
    lowered = stem.lower()
    marker = "_cap"
    if marker in lowered:
        index = lowered.index(marker)
        return stem[:index] or "Referencia"
    return "Referencia"


def count_archive_pages(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(
            1
            for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        )


def discover_archives(directory: Path) -> dict[int, Path]:
    archives: dict[int, Path] = {}
    if not directory.exists():
        return archives

    for path in sorted(directory.glob("*.cbz")):
        chapter_number = parse_chapter_number(path.name)
        if chapter_number is None:
            continue
        archives[chapter_number] = path
    return archives


def pair_chapters(source_dir: Path, reference_dir: Path) -> list[ChapterPair]:
    source_archives = discover_archives(source_dir)
    reference_archives = discover_archives(reference_dir)
    pairs: list[ChapterPair] = []

    for chapter_number, source_path in source_archives.items():
        reference_path = reference_archives.get(chapter_number)
        if reference_path is None:
            continue

        pairs.append(
            ChapterPair(
                chapter_number=chapter_number,
                source_path=str(source_path),
                reference_path=str(reference_path),
                source_pages=count_archive_pages(source_path),
                reference_pages=count_archive_pages(reference_path),
                reference_group=extract_reference_group(reference_path.name),
            )
        )

    return sorted(pairs, key=lambda pair: pair.chapter_number)
