import json
import tempfile
import unittest
from pathlib import Path

from corpus.runtime import build_corpus_memory_map, load_corpus_bundle, slugify_work_title


class CorpusRuntimeTests(unittest.TestCase):
    def test_slugify_work_title_matches_corpus_folder(self):
        self.assertEqual(
            slugify_work_title("The Regressed Mercenary Has a Plan"),
            "the-regressed-mercenary-has-a-plan",
        )

    def test_load_corpus_bundle_reads_benchmark_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "the-regressed-mercenary-has-a-plan"
            corpus_dir.mkdir(parents=True)
            (corpus_dir / "translation_memory_candidates.json").write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "source_text": "Do you think",
                                "target_text": "Voce acha que",
                                "occurrences": 2,
                                "mean_position_delta": 0.01,
                                "source_tokens": 3,
                                "target_tokens": 3,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (corpus_dir / "visual_benchmark_profile.json").write_text(
                json.dumps({"page_geometry": {"median_width": 800}}),
                encoding="utf-8",
            )

            bundle = load_corpus_bundle(
                "The Regressed Mercenary Has a Plan",
                models_root=Path(tmp),
            )

            self.assertEqual(bundle["slug"], "the-regressed-mercenary-has-a-plan")
            self.assertEqual(bundle["visual_benchmark_profile"]["page_geometry"]["median_width"], 800)
            self.assertEqual(bundle["translation_memory_candidates"]["candidates"][0]["source_text"], "Do you think")

    def test_load_corpus_bundle_falls_back_when_models_root_has_no_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            fallback_dir = Path(tmp) / "fallback"
            corpus_dir = fallback_dir / "the-regressed-mercenary-has-a-plan"
            corpus_dir.mkdir(parents=True)
            (corpus_dir / "work_profile.json").write_text(
                json.dumps({"chapter_range": [1, 83]}),
                encoding="utf-8",
            )

            bundle = load_corpus_bundle(
                "The Regressed Mercenary Has a Plan",
                models_root=Path(tmp) / "missing",
                fallback_root=fallback_dir,
            )

            self.assertTrue(bundle["available"])
            self.assertEqual(bundle["work_profile"]["chapter_range"], [1, 83])

    def test_build_corpus_memory_map_keeps_only_high_trust_candidates(self):
        memory_map = build_corpus_memory_map(
            {
                "candidates": [
                    {
                        "source_text": "Do you think",
                        "target_text": "Voce acha que",
                        "occurrences": 2,
                        "mean_position_delta": 0.01,
                        "source_tokens": 3,
                        "target_tokens": 3,
                    },
                    {
                        "source_text": "BAD OCR1",
                        "target_text": "Algo estranho",
                        "occurrences": 1,
                        "mean_position_delta": 0.2,
                        "source_tokens": 2,
                        "target_tokens": 2,
                    },
                ]
            }
        )

        self.assertEqual(memory_map["Do you think"], "Voce acha que")
        self.assertNotIn("BAD OCR1", memory_map)


if __name__ == "__main__":
    unittest.main()
