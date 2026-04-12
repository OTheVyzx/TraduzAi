from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from lab.benchmarking import aggregate_benchmark_results, benchmark_chapter_output


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 1200), color=color).save(path)


def _write_reference_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for index in range(2):
            temp_image = path.parent / f"temp-{index}.jpg"
            _write_image(temp_image, (220, 220, 220))
            archive.write(temp_image, arcname=f"{index + 1:03d}.jpg")
            temp_image.unlink()


def _sample_project() -> dict:
    return {
        "paginas": [
            {
                "numero": 1,
                "arquivo_traduzido": "translated/001.jpg",
                "textos": [
                    {
                        "original": "hello knight",
                        "traduzido": "ola cavaleiro",
                        "confianca_ocr": 0.94,
                        "bbox": [0, 0, 320, 140],
                        "estilo": {"tamanho": 18},
                    },
                    {
                        "original": "hello knight",
                        "traduzido": "ola cavaleiro",
                        "confianca_ocr": 0.91,
                        "bbox": [0, 0, 320, 140],
                        "estilo": {"tamanho": 18},
                    },
                ],
            }
        ]
    }


class BenchmarkingTests(unittest.TestCase):
    def test_benchmark_generates_real_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translated_dir = output_dir / "translated"
            _write_image(translated_dir / "001.jpg", (215, 215, 215))
            (output_dir / "project.json").write_text(
                json.dumps(_sample_project(), ensure_ascii=False),
                encoding="utf-8",
            )

            source_archive = root / "source.cbz"
            reference_archive = root / "reference.cbz"
            _write_reference_archive(source_archive)
            _write_reference_archive(reference_archive)

            textual_profile = {
                "en_stats": {"mean_regions_per_page": 4.4, "mean_chars_per_region": 10.0},
                "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
                "paired_text_stats": {"mean_translation_length_ratio": 1.15},
            }
            visual_profile = {
                "page_geometry": {"median_width": 800, "median_height": 1200, "median_aspect_ratio": 0.666},
                "luminance_profile": {"mean_luminance": 220.0},
            }

            result = benchmark_chapter_output(
                output_dir=output_dir,
                source_archive=source_archive,
                reference_archive=reference_archive,
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )

            self.assertGreater(result.metrics.textual_similarity, 0.0)
            self.assertGreater(result.metrics.term_consistency, 90.0)
            self.assertGreater(result.score_after, result.score_before)

    def test_aggregate_benchmark_averages_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translated_dir = output_dir / "translated"
            _write_image(translated_dir / "001.jpg", (215, 215, 215))
            (output_dir / "project.json").write_text(
                json.dumps(_sample_project(), ensure_ascii=False),
                encoding="utf-8",
            )

            source_archive = root / "source.cbz"
            reference_archive = root / "reference.cbz"
            _write_reference_archive(source_archive)
            _write_reference_archive(reference_archive)

            textual_profile = {
                "en_stats": {"mean_regions_per_page": 4.4, "mean_chars_per_region": 10.0},
                "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
                "paired_text_stats": {"mean_translation_length_ratio": 1.15},
            }
            visual_profile = {
                "page_geometry": {"median_width": 800, "median_height": 1200, "median_aspect_ratio": 0.666},
                "luminance_profile": {"mean_luminance": 220.0},
            }

            single = benchmark_chapter_output(
                output_dir=output_dir,
                source_archive=source_archive,
                reference_archive=reference_archive,
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )
            aggregate = aggregate_benchmark_results([single, single])

            self.assertAlmostEqual(aggregate.score_after, single.score_after, places=1)
            self.assertAlmostEqual(
                aggregate.metrics.term_consistency,
                single.metrics.term_consistency,
                places=1,
            )


if __name__ == "__main__":
    unittest.main()
