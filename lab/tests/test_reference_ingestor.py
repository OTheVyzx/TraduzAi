import tempfile
import unittest
import zipfile
from pathlib import Path

from lab.reference_ingestor import pair_chapters, parse_chapter_number


def write_cbz(path: Path, image_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for image_name in image_names:
            archive.writestr(image_name, b"fake-image")


class ReferenceIngestorTests(unittest.TestCase):
    def test_parse_chapter_number_handles_english_and_ptbr_names(self) -> None:
        self.assertEqual(parse_chapter_number("Chapter 10_5b99a4.cbz"), 10)
        self.assertEqual(parse_chapter_number("ArinVale_Capítulo 10_e04b44.cbz"), 10)
        self.assertEqual(parse_chapter_number("WorldScan_Capítulo 80_e73190.cbz"), 80)

    def test_pair_chapters_joins_by_chapter_number_and_counts_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "exemploen"
            reference_dir = root / "exemploptbr"

            write_cbz(source_dir / "Chapter 10_5b99a4.cbz", ["001.jpg", "002.jpg"])
            write_cbz(source_dir / "Chapter 11_587299.cbz", ["001.jpg"])
            write_cbz(reference_dir / "ArinVale_Capítulo 10_e04b44.cbz", ["001.jpg", "002.jpg"])
            write_cbz(reference_dir / "ArinVale_Capítulo 11_f99743.cbz", ["001.jpg"])

            pairs = pair_chapters(source_dir, reference_dir)

            self.assertEqual([pair.chapter_number for pair in pairs], [10, 11])
            self.assertEqual(pairs[0].source_pages, 2)
            self.assertEqual(pairs[0].reference_pages, 2)
            self.assertEqual(pairs[0].reference_group, "ArinVale")


if __name__ == "__main__":
    unittest.main()
