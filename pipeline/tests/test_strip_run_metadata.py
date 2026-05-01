"""Regressão: garante que `run_chapter` produz metadados com bbox real."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np


class RunChapterMetadataTests(unittest.TestCase):
    def _make_pages(self, tmp_path: Path, n: int = 3):
        """Cria n imagens JPG 200x300 em tmp_path."""
        extraction = tmp_path / "in"
        extraction.mkdir()
        for i in range(n):
            cv2.imwrite(
                str(extraction / f"p{i:02d}.jpg"),
                np.full((300, 200, 3), 200, dtype=np.uint8),
            )
        return sorted(extraction.glob("*.jpg"))

    def _make_mocks(self):
        """Mocks mínimos que devolvem 1 texto com bbox real."""
        class FakeBlock:
            def __init__(self, x1, y1, x2, y2, c=0.9):
                self.x1, self.y1, self.x2, self.y2, self.confidence = x1, y1, x2, y2, c

        detector = MagicMock()
        detector.detect.return_value = [FakeBlock(50, 50, 150, 150)]

        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{
                "id": "t1",
                "bbox": [50, 50, 150, 150],
                "balloon_bbox": [50, 50, 150, 150],
                "text": "HELLO",
                "tipo": "fala",
            }],
            "_vision_blocks": [{"bbox": [50, 50, 150, 150], "confidence": 0.9}],
        }

        translator = MagicMock()
        translator.translate_pages.return_value = [{
            "texts": [{
                "id": "t1",
                "bbox": [50, 50, 150, 150],
                "balloon_bbox": [50, 50, 150, 150],
                "translated": "OLÁ",
                "tipo": "fala",
            }],
            "_vision_blocks": [{"bbox": [50, 50, 150, 150], "confidence": 0.9}],
        }]

        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()

        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()

        return detector, runtime, translator, inpainter, typesetter

    def test_translated_texts_keep_real_bbox_in_output_pages(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output = tmp / "out"
            output.mkdir()

            files = self._make_pages(tmp)
            detector, runtime, translator, inpainter, typesetter = self._make_mocks()

            pages = run_chapter(
                image_files=files,
                output_dir=output,
                target_count=3,
                detector=detector,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
            )

            self.assertEqual(len(pages), 3)

            non_empty = [
                p for p in pages
                if p.text_layers and p.text_layers.get("texts")
            ]
            # Pelo menos 1 página tem o texto
            self.assertGreaterEqual(len(non_empty), 1, "Nenhuma página contém textos")

            # bbox não pode ser o placeholder [0,0,32,32] nem ter dimensão zero
            for p in non_empty:
                for txt in p.text_layers["texts"]:
                    bx1, by1, bx2, by2 = txt["bbox"]
                    self.assertFalse(
                        (bx1, by1, bx2, by2) == (0, 0, 32, 32),
                        f"Page {p.y_top}: bbox é o placeholder [0,0,32,32]",
                    )
                    self.assertGreater(bx2 - bx1, 5, "bbox com largura quase zero")
                    self.assertGreater(by2 - by1, 5, "bbox com altura quase zero")

    def test_output_pages_have_full_schema(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output = tmp / "out"
            output.mkdir()

            files = self._make_pages(tmp)
            detector, runtime, translator, inpainter, typesetter = self._make_mocks()

            pages = run_chapter(
                image_files=files,
                output_dir=output,
                target_count=3,
                detector=detector,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
            )

            for p in pages:
                self.assertIsNotNone(p.page_profile, "page_profile deve estar presente")
                self.assertIn("width", p.page_profile)
                self.assertIn("height", p.page_profile)
                self.assertIn("y_in_strip_top", p.page_profile)
                self.assertIn("y_in_strip_bottom", p.page_profile)
                self.assertIsNotNone(p.inpaint_blocks, "inpaint_blocks deve estar presente")
                self.assertIsInstance(p.inpaint_blocks, list)

    def test_no_bbox_less_text_survives_in_output(self):
        """Textos sem 'bbox' definido não devem aparecer no output final."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output = tmp / "out"
            output.mkdir()

            files = self._make_pages(tmp)

            class FakeBlock:
                def __init__(self):
                    self.x1, self.y1, self.x2, self.y2, self.confidence = 50, 50, 150, 150, 0.9

            detector = MagicMock()
            detector.detect.return_value = [FakeBlock()]

            runtime = MagicMock()
            runtime.run_ocr_stage.return_value = {
                "texts": [
                    {"id": "t1", "bbox": [50, 50, 150, 150], "texto": "A", "tipo": "fala"},
                    {"id": "t2", "texto": "B sem bbox", "tipo": "fala"},  # SEM bbox
                ],
                "_vision_blocks": [{"bbox": [50, 50, 150, 150], "confidence": 0.9}],
            }

            translator = MagicMock()
            translator.translate_pages.return_value = [{
                "texts": [
                    {"id": "t1", "bbox": [50, 50, 150, 150], "translated": "X", "tipo": "fala"},
                    {"id": "t2", "translated": "Y sem bbox", "tipo": "fala"},  # SEM bbox
                ],
                "_vision_blocks": [],
            }]

            inpainter = MagicMock()
            inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
            typesetter = MagicMock()
            typesetter.render_band_image.side_effect = lambda img, _: img.copy()

            pages = run_chapter(
                image_files=files,
                output_dir=output,
                target_count=3,
                detector=detector,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
            )

            for p in pages:
                texts = p.text_layers.get("texts", []) if p.text_layers else []
                for txt in texts:
                    self.assertIn("bbox", txt, "Texto sem bbox sobreviveu no output")
