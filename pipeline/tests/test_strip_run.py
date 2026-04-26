"""Smoke test do entry-point run_chapter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np


class RunChapterSmokeTests(unittest.TestCase):
    def test_run_chapter_produces_target_count_pages(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            extraction = tmp_path / "extracted"
            extraction.mkdir()
            output = tmp_path / "out"
            output.mkdir()

            # 3 páginas de 200x300 preenchidas com cinza
            for i in range(3):
                img = np.full((300, 200, 3), 128, dtype=np.uint8)
                cv2.imwrite(str(extraction / f"p{i:02d}.jpg"), img)

            # Stages totalmente mockadas
            detector = MagicMock()
            detector.detect.return_value = []  # sem balões -> bandas vazias
            runtime = MagicMock()
            translator = MagicMock()
            inpainter = MagicMock()
            typesetter = MagicMock()

            files = sorted(extraction.glob("*.jpg"))
            output_pages = run_chapter(
                image_files=files,
                output_dir=output,
                target_count=5,
                detector=detector,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
            )

            self.assertEqual(len(output_pages), 5)
            # Arquivos foram salvos
            jpgs = sorted(output.glob("*.jpg"))
            self.assertEqual(len(jpgs), 5)
