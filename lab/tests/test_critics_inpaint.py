from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from lab.critics.inpaint_critic import InpaintCritic


def _make_artifact(pages: list[dict], page_images: list[tuple[str, Image.Image]]) -> dict:
    """Monta um artefato fake com output_dir/translated/<arquivos>."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="inpaint_critic_test_"))
    translated_dir = tmp_dir / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)
    for name, image in page_images:
        image.save(translated_dir / name)

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


def _clean_white_image() -> Image.Image:
    return Image.new("RGB", (400, 600), color=(255, 255, 255))


def _image_with_residual_text() -> Image.Image:
    """Gera uma imagem branca com texto residual ruidoso dentro do bbox."""
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Pinta ruido aleatorio pesado na regiao do bbox
    for x in range(50, 150, 2):
        for y in range(80, 180, 2):
            draw.point((x, y), fill=(0, 0, 0))
    for x in range(60, 140, 3):
        draw.line((x, 90, x + 5, 170), fill=(40, 40, 40), width=2)
    return img


class InpaintCriticTests(unittest.TestCase):
    def test_clean_balloon_emits_nothing(self) -> None:
        artifact = _make_artifact(
            pages=[
                {
                    "arquivo_traduzido": "page-01.jpg",
                    "textos": [
                        {
                            "traduzido": "oi",
                            "bbox": [50, 80, 150, 180],
                        }
                    ],
                }
            ],
            page_images=[("page-01.jpg", _clean_white_image())],
        )
        findings = InpaintCritic().analyze(artifact)
        residual = [f for f in findings if f.issue_type == "residual_text_in_balloon"]
        self.assertFalse(residual)

    def test_residual_text_detected_as_high_variance(self) -> None:
        artifact = _make_artifact(
            pages=[
                {
                    "arquivo_traduzido": "page-01.jpg",
                    "textos": [
                        {
                            "traduzido": "texto",
                            "bbox": [50, 80, 150, 180],
                        }
                    ],
                }
            ],
            page_images=[("page-01.jpg", _image_with_residual_text())],
        )
        findings = InpaintCritic().analyze(artifact)
        residual = [f for f in findings if f.issue_type == "residual_text_in_balloon"]
        self.assertTrue(residual)
        self.assertGreater(residual[0].evidence["variance"], 2200)

    def test_missing_output_dir_returns_empty(self) -> None:
        artifact = {
            "chapter_number": 1,
            "project_json": "",
            "output_dir": "/path/that/does/not/exist/xyz123",
        }
        self.assertEqual(InpaintCritic().analyze(artifact), [])

    def test_invalid_bbox_skipped(self) -> None:
        artifact = _make_artifact(
            pages=[
                {
                    "arquivo_traduzido": "page-01.jpg",
                    "textos": [
                        {"traduzido": "hi", "bbox": [0, 0]},
                        {"traduzido": "", "bbox": [10, 10, 50, 50]},  # texto vazio
                    ],
                }
            ],
            page_images=[("page-01.jpg", _clean_white_image())],
        )
        # Nao deve crashar em bbox invalido nem em texto vazio
        findings = InpaintCritic().analyze(artifact)
        self.assertIsInstance(findings, list)


if __name__ == "__main__":
    unittest.main()
