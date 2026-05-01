from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lab.critics.typeset_critic import TypesetCritic


def _make_artifact(pages: list[dict]) -> dict:
    tmp_dir = Path(tempfile.mkdtemp(prefix="typeset_critic_test_"))
    project_json_path = tmp_dir / "project.json"
    project_json_path.write_text(
        json.dumps({"paginas": pages}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "chapter_number": 1,
        "project_json": str(project_json_path),
        "output_dir": str(tmp_dir),
        "source_path": "",
        "reference_path": "",
        "benchmark": {},
    }


class TypesetCriticTests(unittest.TestCase):
    def test_occupancy_too_low(self) -> None:
        # Balao grande, texto curto, fonte pequena → occupancy << 18%
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": "oi",
                            "bbox": [0, 0, 800, 600],
                            "estilo": {"tamanho": 12},
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        low = [f for f in findings if f.issue_type == "occupancy_too_low"]
        self.assertTrue(low)
        self.assertLess(low[0].evidence["occupancy"], 0.18)

    def test_occupancy_too_high(self) -> None:
        # Balao pequeno, texto longo, fonte grande → occupancy > 72%
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": "palavra " * 30,
                            "bbox": [0, 0, 60, 60],
                            "estilo": {"tamanho": 18},
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        high = [f for f in findings if f.issue_type == "occupancy_too_high"]
        self.assertTrue(high)
        self.assertGreater(high[0].evidence["occupancy"], 0.72)

    def test_font_too_small_for_big_balloon(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": "texto razoavel",
                            "bbox": [0, 0, 200, 120],  # altura 120 >> 50
                            "estilo": {"tamanho": 10},  # <14
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        tiny = [f for f in findings if f.issue_type == "font_too_small_for_balloon"]
        self.assertTrue(tiny)
        self.assertLess(tiny[0].evidence["font_size"], 14.0)

    def test_bbox_overflow(self) -> None:
        # Palavra única sem espaços (60 chars): não pode ser wrapped → overflow real.
        # chars_per_line = int(100 / (20*0.55)) = 9; palavra é 60 > 9 → persiste na linha.
        # est_width = 60 * 20 * 0.55 = 660px > 100 * 1.15 = 115px → overflow flagado.
        long_text = "x" * 60
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": long_text,
                            "bbox": [0, 0, 100, 40],
                            "estilo": {"tamanho": 20},
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        overflow = [f for f in findings if f.issue_type == "text_overflow_bbox"]
        self.assertTrue(overflow)
        self.assertEqual(overflow[0].severity, "error")

    def test_long_multiword_text_wraps_to_fit(self) -> None:
        # Texto longo com palavras curtas que o renderer WRAP em múltiplas linhas.
        # Antes do fix, o critic usava len(texto_completo)*size*0.55 e gerava falso positivo.
        # Após o fix, simula word-wrap e percebe que cada linha cabe no bbox.
        # bbox_width=400, font=20 → chars_per_line = int(400/(20*0.55)) = 36
        # "um dois tres quatro cinco seis sete" (35 chars) → cabe em 1 linha ≤ 36 → no overflow
        long_multiword = "um dois tres quatro cinco seis sete"  # 35 chars, 7 words
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": long_multiword,
                            "bbox": [0, 0, 400, 80],
                            "estilo": {"tamanho": 20},
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        overflow = [f for f in findings if f.issue_type == "text_overflow_bbox"]
        self.assertFalse(overflow, "Texto que cabe via wrap não deve ser flagado como overflow")

    def test_connected_lobe_imbalance(self) -> None:
        # Grupo balão com 2 lobos, um com 1 palavra, outro com 10 → ratio 10x
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": "hi",
                            "bbox": [0, 0, 80, 40],
                            "estilo": {"tamanho": 16},
                            "grupo_balao": "g1",
                        },
                        {
                            "traduzido": "one two three four five six seven eight nine ten",
                            "bbox": [100, 0, 400, 100],
                            "estilo": {"tamanho": 16},
                            "grupo_balao": "g1",
                        },
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        imbalance = [f for f in findings if f.issue_type == "connected_lobe_imbalance"]
        self.assertTrue(imbalance)
        self.assertGreaterEqual(imbalance[0].evidence["ratio"], 2.5)

    def test_balanced_layout_emits_nothing(self) -> None:
        artifact = _make_artifact(
            [
                {
                    "textos": [
                        {
                            "traduzido": "texto medio com tamanho razoavel",
                            "bbox": [0, 0, 300, 120],
                            "estilo": {"tamanho": 16},
                        }
                    ]
                }
            ]
        )
        findings = TypesetCritic().analyze(artifact)
        # Pode emitir occupancy_too_low OK; o importante e nao emitir overflow/tiny
        overflow = [f for f in findings if f.issue_type == "text_overflow_bbox"]
        tiny = [f for f in findings if f.issue_type == "font_too_small_for_balloon"]
        self.assertFalse(overflow)
        self.assertFalse(tiny)


if __name__ == "__main__":
    unittest.main()
