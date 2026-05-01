from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.critics.ocr_critic import OcrCritic


def _make_artifact(pages: list[dict]) -> dict:
    """Cria um artefato temporario com project.json minimo para teste."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_critic_test_"))
    project_json_path = tmp_dir / "project.json"
    project_json_path.write_text(
        json.dumps({"paginas": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "chapter_number": 7,
        "project_json": str(project_json_path),
        "output_dir": str(tmp_dir),
        "source_path": "",
        "reference_path": "",
        "benchmark": {},
    }


class OcrCriticTests(unittest.TestCase):
    def test_low_confidence_emits_finding(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "LOREM IPSUM DOLOR",
                            "confianca_ocr": 0.35,
                            "bbox": [10, 20, 120, 60],
                        }
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].issue_type, "low_confidence")
        self.assertEqual(findings[0].severity, "error")  # <0.4 => error
        self.assertAlmostEqual(findings[0].evidence["confidence"], 0.35, places=3)

    def test_low_confidence_warning_tier(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "HELLO WORLD",
                            "confianca_ocr": 0.5,
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "warning")

    def test_confidence_above_threshold_emits_nothing(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "HIGH CONFIDENCE",
                            "confianca_ocr": 0.95,
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        self.assertEqual(findings, [])

    def test_watermark_leaked(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "visit asura scans for more",
                            "confianca_ocr": 0.9,
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        watermark = [f for f in findings if f.issue_type == "watermark_leaked"]
        self.assertTrue(watermark)
        self.assertIn(watermark[0].evidence["matched_token"], {"asura", "scans"})
        self.assertEqual(watermark[0].severity, "error")

    def test_repeated_digits_artifact(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "text with 1111111 garbage",
                            "confianca_ocr": 0.9,
                            "bbox": [0, 0, 100, 50],
                        }
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        repeated = [f for f in findings if f.issue_type == "ocr_artifact_repeated_digits"]
        self.assertTrue(repeated)

    def test_duplicated_boxes_high_iou(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "original": "linha A",
                            "confianca_ocr": 0.9,
                            "bbox": [0, 0, 100, 50],
                        },
                        {
                            "original": "linha A duplicada",
                            "confianca_ocr": 0.9,
                            "bbox": [10, 5, 105, 55],
                        },
                    ]
                }
            ]
        )
        findings = OcrCritic().analyze(artifact)
        dupes = [f for f in findings if f.issue_type == "duplicated_ocr_box"]
        self.assertTrue(dupes)
        self.assertGreaterEqual(dupes[0].evidence["iou"], 0.3)

    def test_empty_pages_yields_no_findings(self) -> None:
        artifact = _make_artifact([])
        self.assertEqual(OcrCritic().analyze(artifact), [])


if __name__ == "__main__":
    unittest.main()
