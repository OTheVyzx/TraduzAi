"""Smoke test do entry-point run_chapter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


def _make_detector_with_n_balloons(n: int, page_height: int = 300, page_width: int = 200):
    """Retorna mock detector com n balões bem separados (um por página)."""
    detector = MagicMock()

    def make_block(y1, y2):
        b = MagicMock()
        b.x1 = 10.0; b.y1 = float(y1)
        b.x2 = float(page_width - 15); b.y2 = float(y2)
        b.confidence = 0.9
        return b

    # Balloon height = 50px; even for n=1 (strip=300) cap_h=75 > 50 → not oversized
    blocks = [make_block(i * page_height + 10, i * page_height + 60) for i in range(n)]
    detector.detect.return_value = blocks
    return detector


def _write_pages(tmp_path: Path, n: int, page_height: int = 300, page_width: int = 200) -> list:
    paths = []
    for i in range(n):
        img = np.full((page_height, page_width, 3), 128, dtype=np.uint8)
        p = tmp_path / f"p{i:02d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return sorted(paths)


def _fake_process_band_factory(records: list, ocr_extras: dict | None = None):
    """Retorna side_effect que salva args e define ocr_result mínimo."""
    call_index = [0]

    def fake_pb(band, **kw):
        idx = call_index[0]
        records.append({
            "band_history": list(kw.get("band_history") or []),
            "glossario": dict(kw.get("glossario") or {}),
        })
        band.rendered_slice = band.strip_slice.copy()
        extras = (ocr_extras or {}).get(idx, {})
        band.ocr_result = {"texts": [], "_vision_blocks": [], **extras}
        call_index[0] += 1
        return band

    return fake_pb


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


class RunningHistoryTests(unittest.TestCase):
    """H.4 — history rolante de bandas passado para contextual reviewer."""

    def test_band_history_empty_for_first_band(self):
        """Primeira banda recebe history vazio."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 3)
            output = tmp_path / "out"

            records = []
            with patch("strip.run.process_band", side_effect=_fake_process_band_factory(records)):
                run_chapter(
                    image_files=files, output_dir=output, target_count=3,
                    detector=_make_detector_with_n_balloons(3),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                )

            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[0]["band_history"], [])

    def test_band_history_grows_by_one_each_band(self):
        """Banda N recebe exatamente N entradas no history."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 3)
            output = tmp_path / "out"

            records = []
            with patch("strip.run.process_band", side_effect=_fake_process_band_factory(records)):
                run_chapter(
                    image_files=files, output_dir=output, target_count=3,
                    detector=_make_detector_with_n_balloons(3),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                )

            self.assertEqual(len(records), 3)
            self.assertEqual(len(records[0]["band_history"]), 0)
            self.assertEqual(len(records[1]["band_history"]), 1)
            self.assertEqual(len(records[2]["band_history"]), 2)

    def test_band_history_contains_previous_ocr_result(self):
        """History de banda 2 contém o ocr_result da banda 1."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"

            # Banda 0 terá _marker no ocr_result
            ocr_extras = {0: {"_marker": "banda_zero"}}
            records = []
            with patch("strip.run.process_band",
                       side_effect=_fake_process_band_factory(records, ocr_extras)):
                run_chapter(
                    image_files=files, output_dir=output, target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                )

            self.assertEqual(len(records), 2)
            # banda 1 deve ver o ocr_result da banda 0 no history
            hist = records[1]["band_history"]
            self.assertEqual(len(hist), 1)
            self.assertEqual(hist[0].get("_marker"), "banda_zero")


class RunningGlossaryTests(unittest.TestCase):
    """H.3 — glossário mutável acumulado entre bandas."""

    def test_initial_glossary_passed_to_first_band(self):
        """O glossário inicial é passado à primeira banda."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"

            records = []
            with patch("strip.run.process_band", side_effect=_fake_process_band_factory(records)):
                run_chapter(
                    image_files=files, output_dir=output, target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                    glossario={"HERO": "herói"},
                )

            self.assertEqual(records[0]["glossario"].get("HERO"), "herói")

    def test_glossary_additions_propagate_to_next_band(self):
        """_glossary_additions de banda 0 aparecem no glossário de banda 1."""
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"

            # Banda 0 expõe adição ao glossário
            ocr_extras = {0: {"_glossary_additions": {"FENRIS": "Fenris"}}}
            records = []
            with patch("strip.run.process_band",
                       side_effect=_fake_process_band_factory(records, ocr_extras)):
                run_chapter(
                    image_files=files, output_dir=output, target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                    glossario={"BASE": "base"},
                )

            # Banda 0: ainda não tem FENRIS
            self.assertNotIn("FENRIS", records[0]["glossario"])
            self.assertIn("BASE", records[0]["glossario"])
            # Banda 1: deve ter FENRIS que banda 0 adicionou
            self.assertIn("FENRIS", records[1]["glossario"])
            self.assertEqual(records[1]["glossario"]["FENRIS"], "Fenris")
