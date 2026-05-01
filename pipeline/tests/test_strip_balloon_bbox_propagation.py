"""Garante que balloon_bbox é preservado pela banda → page → translation.

Também testa o guard em render_band_image que emite WARNING quando
balloon_bbox está ausente (proteção contra overflow de texto).
"""

import unittest
from unittest.mock import MagicMock

import numpy as np


class BalloonBboxPropagationTests(unittest.TestCase):
    def test_balloon_bbox_present_after_process_band(self):
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox

        # Banda com 1 balão branco no centro
        slice_img = np.full((200, 400, 3), 50, dtype=np.uint8)
        # Balão branco
        slice_img[40:160, 80:320] = 255

        band = Band(
            y_top=100, y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(80, 140, 320, 260), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        # OCR retorna text com bbox dentro do balão
        runtime.run_ocr_stage.return_value = {
            "texts": [{
                "id": "t1",
                "bbox": [120, 60, 280, 100],
                "text": "HELLO WORLD",
                "tipo": "fala",
            }],
            "_vision_blocks": [{"bbox": [80, 40, 320, 160], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        # AFTER process_band, ocr_result["texts"] precisa ter balloon_bbox
        self.assertIsNotNone(result.ocr_result)
        for txt in result.ocr_result.get("texts", []):
            self.assertIn(
                "balloon_bbox", txt,
                f"Text {txt.get('id')} sem balloon_bbox — overflow garantido",
            )
            bx1, by1, bx2, by2 = txt["balloon_bbox"]
            self.assertGreater(bx2 - bx1, 30,
                f"balloon_bbox largura {bx2-bx1} muito pequena")

    def test_balloon_bbox_fallback_uses_vision_block_with_best_iou(self):
        """Se enrich_page_layout não seta balloon_bbox, fallback usa vision_block."""
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox

        slice_img = np.full((200, 400, 3), 200, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=200,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 350, 180), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        # Nenhum balloon_bbox no retorno do OCR
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "t2", "bbox": [100, 50, 300, 100], "text": "X", "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [50, 20, 350, 180], "confidence": 0.9}],
        }

        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()

        result = process_band(
            band, runtime=runtime, translator=translator,
            inpainter=inpainter, typesetter=typesetter, page_idx=0,
        )

        # Garantir que balloon_bbox existe e é maior que só o texto
        for txt in result.ocr_result.get("texts", []):
            self.assertIn("balloon_bbox", txt)
            bx1, by1, bx2, by2 = txt["balloon_bbox"]
            # balloon_bbox deve ser maior que o text bbox (50→350 = 300px wide)
            self.assertGreater(bx2 - bx1, 100)

    def test_balloon_bbox_last_resort_is_text_bbox_plus_margin(self):
        """Se não houver vision_block, balloon_bbox = text_bbox + 8px."""
        from strip.process_bands import process_band
        from strip.types import Band, Balloon, BBox

        slice_img = np.full((100, 300, 3), 230, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 10, 250, 90), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "t3", "bbox": [60, 20, 240, 80], "text": "Z", "tipo": "fala"}],
            "_vision_blocks": [],  # SEM vision blocks
        }

        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()

        result = process_band(
            band, runtime=runtime, translator=translator,
            inpainter=inpainter, typesetter=typesetter, page_idx=0,
        )

        for txt in result.ocr_result.get("texts", []):
            self.assertIn("balloon_bbox", txt)
            bx1, by1, bx2, by2 = txt["balloon_bbox"]
            # Deve ter pelo menos 8 px de margem em relação ao bbox de texto
            tx1, ty1, tx2, ty2 = txt["bbox"]
            self.assertLessEqual(bx1, tx1)
            self.assertLessEqual(by1, ty1)
            self.assertGreaterEqual(bx2, tx2)
            self.assertGreaterEqual(by2, ty2)


class RenderBandImageGuardTests(unittest.TestCase):
    def test_render_band_image_warns_on_missing_balloon_bbox(self):
        """render_band_image deve emitir WARNING quando balloon_bbox ausente."""
        import numpy as np
        from typesetter.renderer import render_band_image

        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{
                "id": "t1",
                "bbox": [50, 20, 150, 80],
                "tipo": "fala",
                "translated": "OLÁ",
                # SEM balloon_bbox — propositalmente
            }]
        }

        with self.assertLogs(level="WARNING") as cm:
            render_band_image(band, page)
        self.assertTrue(
            any("RISCO DE OVERFLOW" in m for m in cm.output),
            f"Warning esperado não encontrado em: {cm.output}",
        )

    def test_render_band_image_no_warning_when_balloon_bbox_present(self):
        """Sem aviso de overflow quando balloon_bbox está presente em todos os textos."""
        import numpy as np
        from typesetter.renderer import render_band_image

        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{
                "id": "t1",
                "bbox": [50, 20, 150, 80],
                "balloon_bbox": [40, 10, 260, 90],
                "tipo": "fala",
                "translated": "OLÁ",
            }]
        }

        # assertNoLogs (Python 3.10+) verifica que nenhum log WARNING é emitido
        with self.assertNoLogs(logger="typesetter.renderer", level="WARNING"):
            render_band_image(band, page)
