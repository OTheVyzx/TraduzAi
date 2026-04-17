from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.critics.translation_critic import TranslationCritic


def _make_artifact(pages: list[dict]) -> dict:
    tmp_dir = Path(tempfile.mkdtemp(prefix="translation_critic_test_"))
    project_json_path = tmp_dir / "project.json"
    project_json_path.write_text(
        json.dumps({"paginas": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "chapter_number": 3,
        "project_json": str(project_json_path),
        "output_dir": str(tmp_dir),
        "source_path": "",
        "reference_path": "",
        "benchmark": {},
    }


class TranslationCriticTests(unittest.TestCase):
    def test_untranslated_same_text(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "The hero strikes back",
                            "traduzido": "The hero strikes back",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        untranslated = [f for f in findings if f.issue_type == "untranslated"]
        self.assertTrue(untranslated)
        self.assertEqual(untranslated[0].severity, "error")

    def test_short_strings_are_ignored(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {"original": "Ok", "traduzido": "Ok", "bbox": [0, 0, 50, 20]},
                        {"original": "!!", "traduzido": "!!", "bbox": [0, 0, 50, 20]},
                    ]
                }
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        untranslated = [f for f in findings if f.issue_type == "untranslated"]
        self.assertFalse(untranslated)

    def test_encoding_artifact_detected(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "You're my friend",
                            "traduzido": "VocÃª eh meu amigo",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        encoding = [f for f in findings if f.issue_type == "encoding_artifact"]
        self.assertTrue(encoding)
        self.assertEqual(encoding[0].severity, "error")

    def test_length_ratio_outlier_too_short(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "This is a long original sentence with meaning",
                            "traduzido": "Oi",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        ratio = [f for f in findings if f.issue_type == "length_ratio_outlier"]
        self.assertTrue(ratio)
        self.assertLess(ratio[0].evidence["ratio"], 0.5)

    def test_term_inconsistency_across_pages(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "Master of Darkness",
                            "traduzido": "Senhor das Trevas",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                },
                {
                    "textos": [
                        {
                            "original": "Master of Darkness",
                            "traduzido": "Mestre da Escuridao",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                },
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        inconsistency = [f for f in findings if f.issue_type == "term_inconsistency"]
        self.assertTrue(inconsistency)
        variants = inconsistency[0].evidence["variants"]
        self.assertGreaterEqual(len(variants), 2)

    def test_clean_translation_emits_nothing(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "Good morning, hero",
                            "traduzido": "Bom dia, heroi",
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = TranslationCritic().analyze(artifact)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
