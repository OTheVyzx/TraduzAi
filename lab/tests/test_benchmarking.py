from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from lab.benchmarking import (
    _after_textual_similarity,
    _layout_occupancy,
    _manual_edits_saved,
    _readability,
    aggregate_benchmark_results,
    benchmark_chapter_output,
)


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
    def test_manual_edits_saved_ignores_noise_and_rewards_strong_dialogue_confidence(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "textos": [
                        {
                            "original": "HELLO THERE",
                            "traduzido": "OLA",
                            "confianca_ocr": 0.94,
                            "bbox": [0, 0, 220, 90],
                            "estilo": {"tamanho": 22},
                        },
                        {
                            "original": "I WON'T LOSE",
                            "traduzido": "NAO VOU PERDER",
                            "confianca_ocr": 0.96,
                            "bbox": [0, 100, 260, 210],
                            "estilo": {"tamanho": 24},
                        },
                        {
                            "original": "ASURASCANS.COM",
                            "traduzido": "ASURASCANS. COM",
                            "confianca_ocr": 0.41,
                            "bbox": [0, 220, 180, 260],
                            "estilo": {"tamanho": 18},
                        },
                    ],
                }
            ]
        }

        score = _manual_edits_saved(project)

        self.assertGreater(score, 95.0)

    def test_layout_occupancy_accounts_for_font_shrinking_when_text_wraps(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "textos": [
                        {
                            "original": "HE BROKE THE MANA-INFUSED BLADE WITH SHEER GRIP STRENGTH.",
                            "traduzido": "ELE QUEBROU A LAMINA INFUNDIDA COM MANA COM PURA FORCA DE PREENSÃO.",
                            "confianca_ocr": 0.92,
                            "bbox": [272, 728, 635, 898],
                            "estilo": {"tamanho": 48},
                        },
                        {
                            "original": "MY ATTACKS CANNOT KEEP UP WITH HIS BLOCKING SPEED!",
                            "traduzido": "MEUS ATAQUES NÃO CONSEGUEM NEM ACOMPANHAR SUA VELOCIDADE DE BLOQUEIO!",
                            "confianca_ocr": 0.88,
                            "bbox": [280, 775, 745, 920],
                            "estilo": {"tamanho": 48},
                        },
                    ],
                }
            ]
        }

        occupancy = _layout_occupancy(project)

        self.assertGreater(occupancy, 70.0)

    def test_readability_does_not_penalize_large_but_legible_fonts(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "textos": [
                        {
                            "original": "HE BROKE THE MANA-INFUSED BLADE WITH SHEER GRIP STRENGTH.",
                            "traduzido": "ELE QUEBROU A LAMINA INFUNDIDA COM MANA COM PURA FORCA DE PREENSÃO.",
                            "confianca_ocr": 0.92,
                            "bbox": [0, 0, 520, 250],
                            "estilo": {"tamanho": 48},
                        },
                        {
                            "original": "DESMOND...",
                            "traduzido": "DESMOND...",
                            "confianca_ocr": 0.88,
                            "bbox": [0, 260, 320, 420],
                            "estilo": {"tamanho": 48},
                        },
                    ],
                }
            ]
        }

        readability = _readability(project)

        self.assertGreater(readability, 70.0)

    def test_visual_cleanup_prefers_corpus_geometry_when_reference_paging_is_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translated_dir = output_dir / "translated"
            _write_image(translated_dir / "001.jpg", (160, 160, 160))
            for index in range(2, 7):
                Image.new("RGB", (800, 2500), color=(160, 160, 160)).save(translated_dir / f"{index:03d}.jpg")
            (output_dir / "project.json").write_text(json.dumps(_sample_project(), ensure_ascii=False), encoding="utf-8")

            source_archive = root / "source.cbz"
            reference_archive = root / "reference.cbz"
            _write_reference_archive(source_archive)

            with zipfile.ZipFile(reference_archive, "w") as archive:
                for index in range(6):
                    temp_image = root / f"ref-{index}.jpg"
                    Image.new("RGB", (800, 1381), color=(160, 160, 160)).save(temp_image)
                    archive.write(temp_image, arcname=f"{index + 1:03d}.jpg")
                    temp_image.unlink()

            textual_profile = {
                "en_stats": {"mean_regions_per_page": 4.4, "mean_chars_per_region": 10.0},
                "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
                "paired_text_stats": {"mean_translation_length_ratio": 1.15},
            }
            visual_profile = {
                "page_geometry": {"median_width": 800, "median_height": 2500, "median_aspect_ratio": 0.32},
                "luminance_profile": {"mean_luminance": 160.0},
            }

            result = benchmark_chapter_output(
                output_dir=output_dir,
                source_archive=source_archive,
                reference_archive=reference_archive,
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )

            self.assertGreater(result.metrics.visual_cleanup, 70.0)

    def test_visual_cleanup_prefers_source_geometry_when_output_matches_source_long_strip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translated_dir = output_dir / "translated"
            translated_dir.mkdir(parents=True, exist_ok=True)
            for index in range(1, 7):
                Image.new("RGB", (800, 2500), color=(160, 160, 160)).save(translated_dir / f"{index:03d}.jpg")
            (output_dir / "project.json").write_text(json.dumps(_sample_project(), ensure_ascii=False), encoding="utf-8")

            source_archive = root / "source.cbz"
            with zipfile.ZipFile(source_archive, "w") as archive:
                for index in range(6):
                    temp_image = root / f"src-{index}.jpg"
                    Image.new("RGB", (800, 2500), color=(160, 160, 160)).save(temp_image)
                    archive.write(temp_image, arcname=f"{index + 1:03d}.jpg")
                    temp_image.unlink()

            reference_archive = root / "reference.cbz"
            with zipfile.ZipFile(reference_archive, "w") as archive:
                for index in range(6):
                    temp_image = root / f"ref-{index}.jpg"
                    Image.new("RGB", (800, 1381), color=(160, 160, 160)).save(temp_image)
                    archive.write(temp_image, arcname=f"{index + 1:03d}.jpg")
                    temp_image.unlink()

            textual_profile = {
                "en_stats": {"mean_regions_per_page": 4.4, "mean_chars_per_region": 10.0},
                "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
                "paired_text_stats": {"mean_translation_length_ratio": 1.15},
            }
            visual_profile = {
                "page_geometry": {"median_width": 800, "median_height": 2500, "median_aspect_ratio": 0.32},
                "luminance_profile": {"mean_luminance": 160.0},
            }

            result = benchmark_chapter_output(
                output_dir=output_dir,
                source_archive=source_archive,
                reference_archive=reference_archive,
                textual_profile=textual_profile,
                visual_profile=visual_profile,
            )

            self.assertGreater(result.metrics.visual_cleanup, 95.0)

    def test_textual_similarity_accepts_source_preserving_ratio_within_chapter_band(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "textos": [
                        {
                            "original": "HE BROKE THE MANA-INFUSED BLADE WITH SHEER GRIP STRENGTH.",
                            "traduzido": "ELE QUEBROU A LAMINA INFUNDIDA COM MANA COM PURA FORCA DE PREENSÃO.",
                            "confianca_ocr": 0.92,
                            "bbox": [0, 0, 520, 250],
                            "estilo": {"tamanho": 28},
                        },
                        {
                            "original": "MY ATTACKS CANNOT KEEP UP WITH HIS BLOCKING SPEED!",
                            "traduzido": "MEUS ATAQUES NÃO CONSEGUEM NEM ACOMPANHAR SUA VELOCIDADE DE BLOQUEIO!",
                            "confianca_ocr": 0.88,
                            "bbox": [0, 260, 520, 440],
                            "estilo": {"tamanho": 28},
                        },
                    ],
                }
            ]
        }
        textual_profile = {
            "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
            "paired_text_stats": {"mean_translation_length_ratio": 1.18},
        }

        similarity = _after_textual_similarity(project, textual_profile)

        self.assertGreater(similarity, 95.0)

    def test_textual_similarity_handles_sparse_action_chapters_using_source_length_signal(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "textos": [
                        {
                            "original": "THE REGRESSED MERCENARY'S MACHINATIONS",
                            "traduzido": "AS MAQUINACOES DO MERCENARIO REGREDIDO",
                            "confianca_ocr": 0.9,
                            "bbox": [0, 0, 320, 140],
                            "estilo": {"tamanho": 26},
                        },
                        {
                            "original": "HE BROKE THE MANA-INFUSED BLADE WITH SHEER GRIP STRENGTH.",
                            "traduzido": "ELE QUEBROU A LAMINA INFUNDIDA COM MANA COM PURA FORCA DE PREENSÃO.",
                            "confianca_ocr": 0.9,
                            "bbox": [0, 150, 420, 320],
                            "estilo": {"tamanho": 24},
                        },
                    ],
                },
                {"numero": 2, "textos": []},
            ]
        }
        textual_profile = {
            "pt_stats": {"mean_regions_per_page": 3.8, "mean_chars_per_region": 12.0},
            "paired_text_stats": {"mean_translation_length_ratio": 1.15},
        }

        similarity = _after_textual_similarity(project, textual_profile)

        self.assertGreater(similarity, 50.0)

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
