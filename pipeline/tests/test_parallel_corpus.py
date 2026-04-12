import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from corpus.parallel_dataset import (
    build_alignment_profile,
    build_manifest,
    build_page_alignment_profile,
    build_quality_profile,
    build_textual_benchmark_profile,
    build_translation_memory_candidates,
    build_visual_benchmark_profile,
    build_work_profile,
    pair_parallel_chapters,
    parse_en_chapter_filename,
    parse_pt_chapter_filename,
)


def _write_cbz(path: Path, page_count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(1, page_count + 1):
            archive.writestr(f"{index:03d}.jpg", b"fake")


def _render_pattern_page(width: int, height: int, gray: int, offset: int) -> bytes:
    image = Image.new("RGB", (width, height), (gray, gray, gray))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40 + offset, 60, 160 + offset, 180), outline=(255, 0, 0), width=8)
    draw.rectangle((220, 200 + offset, 340, 320 + offset), outline=(0, 0, 255), width=8)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_image_cbz(path: Path, pages: list[tuple[int, int, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index, (width, height, gray) in enumerate(pages, start=1):
            image = Image.new("RGB", (width, height), (gray, gray, gray))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            archive.writestr(f"{index:03d}.jpg", buffer.getvalue())


def _write_pattern_cbz(path: Path, specs: list[tuple[int, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index, (gray, offset) in enumerate(specs, start=1):
            archive.writestr(f"{index:03d}.jpg", _render_pattern_page(400, 600, gray, offset))


class ParallelCorpusTests(unittest.TestCase):
    def test_parsers_extract_chapter_and_source(self):
        pt = parse_pt_chapter_filename("ArinVale_Capítulo 82_8f4b5e.cbz")
        en = parse_en_chapter_filename("Chapter 82_5282ef.cbz")

        self.assertEqual(pt["chapter"], 82)
        self.assertEqual(pt["source_group"], "ArinVale")
        self.assertEqual(en["chapter"], 82)

    def test_pairing_matches_chapters(self):
        pt_files = [
            {"chapter": 82, "path": "pt82.cbz", "source_group": "MangaFlix"},
            {"chapter": 81, "path": "pt81.cbz", "source_group": "MangaFlix"},
        ]
        en_files = [
            {"chapter": 81, "path": "en81.cbz"},
            {"chapter": 82, "path": "en82.cbz"},
        ]

        pairs = pair_parallel_chapters(pt_files, en_files)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["chapter"], 81)
        self.assertEqual(pairs[1]["chapter"], 82)

    def test_manifest_and_profiles_include_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pt81 = root / "MangaFlix_Capítulo 81_d776ac.cbz"
            en81 = root / "Chapter 81_952087.cbz"
            _write_cbz(pt81, 12)
            _write_cbz(en81, 11)

            pairs = pair_parallel_chapters(
                [{"chapter": 81, "path": str(pt81), "source_group": "MangaFlix"}],
                [{"chapter": 81, "path": str(en81)}],
            )
            manifest = build_manifest(
                work_slug="the-regressed-mercenary-has-a-plan",
                pt_entries=[{"chapter": 81, "path": str(pt81), "source_group": "MangaFlix"}],
                en_entries=[{"chapter": 81, "path": str(en81)}],
                pairs=pairs,
            )
            quality = build_quality_profile(manifest)
            alignment = build_alignment_profile(manifest)

            self.assertEqual(manifest["paired_chapters"][0]["chapter"], 81)
            self.assertEqual(manifest["paired_chapters"][0]["pt_pages"], 12)
            self.assertEqual(manifest["paired_chapters"][0]["en_pages"], 11)
            self.assertEqual(quality["total_paired_chapters"], 1)
            self.assertEqual(quality["pt_source_distribution"]["MangaFlix"], 1)
            self.assertEqual(alignment["page_deltas"][0]["page_delta"], 1)
            json.dumps(manifest)
            json.dumps(quality)
            json.dumps(alignment)

    def test_work_profile_is_aggregated_by_work_not_by_chapter(self):
        manifest = {
            "work_slug": "the-regressed-mercenary-has-a-plan",
            "total_pt_chapters": 2,
            "total_en_chapters": 2,
            "total_paired_chapters": 2,
            "paired_chapters": [
                {
                    "chapter": 81,
                    "pt_path": "pt81.cbz",
                    "en_path": "en81.cbz",
                    "source_group": "ArinVale",
                    "pt_pages": 12,
                    "en_pages": 11,
                },
                {
                    "chapter": 82,
                    "pt_path": "pt82.cbz",
                    "en_path": "en82.cbz",
                    "source_group": "MangaFlix",
                    "pt_pages": 10,
                    "en_pages": 10,
                },
            ],
        }

        profile = build_work_profile(manifest)

        self.assertEqual(profile["work_slug"], "the-regressed-mercenary-has-a-plan")
        self.assertEqual(profile["chapter_range"], [81, 82])
        self.assertEqual(profile["paired_totals"]["pt_pages"], 22)
        self.assertEqual(profile["paired_totals"]["en_pages"], 21)
        self.assertEqual(profile["paired_totals"]["page_delta"], 1)
        self.assertEqual(profile["provenance"]["pt_source_distribution"]["ArinVale"], 1)
        self.assertNotIn("chapter_profiles", profile)

    def test_visual_benchmark_profile_aggregates_page_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pt81 = root / "ArinVale_Capítulo 81_a.cbz"
            pt82 = root / "MangaFlix_Capítulo 82_b.cbz"
            _write_image_cbz(pt81, [(800, 2400, 250), (800, 2400, 30)])
            _write_image_cbz(pt82, [(900, 2400, 180)])

            manifest = {
                "work_slug": "the-regressed-mercenary-has-a-plan",
                "total_pt_chapters": 2,
                "total_en_chapters": 2,
                "total_paired_chapters": 2,
                "paired_chapters": [
                    {
                        "chapter": 81,
                        "pt_path": str(pt81),
                        "en_path": "en81.cbz",
                        "source_group": "ArinVale",
                        "pt_pages": 2,
                        "en_pages": 2,
                    },
                    {
                        "chapter": 82,
                        "pt_path": str(pt82),
                        "en_path": "en82.cbz",
                        "source_group": "MangaFlix",
                        "pt_pages": 1,
                        "en_pages": 1,
                    },
                ],
            }

            profile = build_visual_benchmark_profile(manifest, max_pages_per_chapter=4)

            self.assertEqual(profile["work_slug"], "the-regressed-mercenary-has-a-plan")
            self.assertEqual(profile["sampled_pages"], 3)
            self.assertEqual(profile["page_geometry"]["median_width"], 800)
            self.assertEqual(profile["page_geometry"]["median_height"], 2400)
            self.assertAlmostEqual(profile["page_geometry"]["median_aspect_ratio"], 0.3333, places=3)
            self.assertEqual(profile["luminance_profile"]["dark_pages"], 1)
            self.assertEqual(profile["luminance_profile"]["light_pages"], 1)
            self.assertEqual(profile["luminance_profile"]["mid_pages"], 1)

    def test_page_alignment_profile_handles_inserted_extra_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pt81 = root / "ArinVale_Capítulo 81_a.cbz"
            en81 = root / "Chapter 81_a.cbz"
            _write_pattern_cbz(pt81, [(240, 0), (140, 30), (60, 60)])
            _write_pattern_cbz(en81, [(240, 0), (210, 90), (140, 30), (60, 60)])

            manifest = {
                "work_slug": "the-regressed-mercenary-has-a-plan",
                "total_pt_chapters": 1,
                "total_en_chapters": 1,
                "total_paired_chapters": 1,
                "paired_chapters": [
                    {
                        "chapter": 81,
                        "pt_path": str(pt81),
                        "en_path": str(en81),
                        "source_group": "ArinVale",
                        "pt_pages": 3,
                        "en_pages": 4,
                    }
                ],
            }

            profile = build_page_alignment_profile(manifest)
            mappings = profile["chapters"][0]["mappings"]

            self.assertEqual([(m["pt_page"], m["en_page"]) for m in mappings], [(1, 1), (2, 3), (3, 4)])
            self.assertGreaterEqual(profile["chapters"][0]["coverage_ratio"], 1.0)

    def test_translation_memory_and_textual_benchmark_use_ocr_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            en_a = root / "en-a.png"
            pt_a = root / "pt-a.png"
            en_b = root / "en-b.png"
            pt_b = root / "pt-b.png"
            for path in [en_a, pt_a, en_b, pt_b]:
                Image.new("RGB", (320, 480), (255, 255, 255)).save(path)

            sample_pairs = [
                {"chapter": 1, "en_image_path": str(en_a), "pt_image_path": str(pt_a)},
                {"chapter": 2, "en_image_path": str(en_b), "pt_image_path": str(pt_b)},
            ]

            ocr_map = {
                str(en_a): {
                    "texts": [
                        {"text": "Hello there", "bbox": [0, 0, 80, 30], "tipo": "fala", "ocr_source": "primary"},
                        {"text": "Mercenary", "bbox": [0, 40, 80, 70], "tipo": "narracao", "ocr_source": "primary"},
                    ]
                },
                str(pt_a): {
                    "texts": [
                        {"text": "Olá aí", "bbox": [0, 0, 80, 30], "tipo": "fala", "ocr_source": "primary"},
                        {"text": "Mercenário", "bbox": [0, 40, 80, 70], "tipo": "narracao", "ocr_source": "primary"},
                    ]
                },
                str(en_b): {
                    "texts": [
                        {"text": "Hello there", "bbox": [0, 0, 80, 30], "tipo": "fala", "ocr_source": "fallback"},
                    ]
                },
                str(pt_b): {
                    "texts": [
                        {"text": "Olá aí", "bbox": [0, 0, 80, 30], "tipo": "fala", "ocr_source": "primary"},
                    ]
                },
            }

            def fake_ocr_runner(image_path: str) -> dict:
                return ocr_map[image_path]

            memory = build_translation_memory_candidates(
                sample_pairs=sample_pairs,
                work_slug="the-regressed-mercenary-has-a-plan",
                ocr_runner=fake_ocr_runner,
            )
            textual = build_textual_benchmark_profile(
                sample_pairs=sample_pairs,
                work_slug="the-regressed-mercenary-has-a-plan",
                ocr_runner=fake_ocr_runner,
            )

            self.assertEqual(memory["sampled_page_pairs"], 2)
            self.assertGreaterEqual(memory["candidate_count"], 2)
            self.assertEqual(memory["candidates"][0]["source_text"], "Hello there")
            self.assertEqual(memory["candidates"][0]["target_text"], "Olá aí")
            self.assertEqual(memory["candidates"][0]["occurrences"], 2)
            self.assertEqual(textual["sampled_page_pairs"], 2)
            self.assertEqual(textual["en_stats"]["ocr_source_distribution"]["primary"], 2)
            self.assertEqual(textual["en_stats"]["ocr_source_distribution"]["fallback"], 1)
            self.assertAlmostEqual(textual["paired_text_stats"]["mean_translation_length_ratio"], 0.73, places=2)


if __name__ == "__main__":
    unittest.main()
