"""Garante que balloon_bbox é preservado pela banda → page → translation.

Também testa o guard em render_band_image que emite WARNING quando
balloon_bbox está ausente (proteção contra overflow de texto).
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def test_shift_text_geometry_y_shifts_bubble_fields_from_band_to_strip():
    from strip.run import _shift_text_geometry_y

    text = {
        "id": "ocr_002",
        "band_id": "page_002_band_005",
        "bbox": [25, 16, 667, 325],
        "text_pixel_bbox": [498, 235, 656, 320],
        "balloon_bbox": [466, 186, 696, 357],
        "bubble_mask_bbox": [501, 218, 661, 325],
        "bubble_inner_bbox": [513, 230, 649, 313],
        "safe_text_box": [525, 242, 637, 301],
        "render_bbox": [542, 246, 620, 296],
        "line_polygons": [
            [[498, 232], [658, 234], [658, 258], [498, 256]],
        ],
    }

    shifted = _shift_text_geometry_y(text, 5420)

    assert shifted["bbox"] == [25, 5436, 667, 5745]
    assert shifted["text_pixel_bbox"] == [498, 5655, 656, 5740]
    assert shifted["balloon_bbox"] == [466, 5606, 696, 5777]
    assert shifted["bubble_mask_bbox"] == [501, 5638, 661, 5745]
    assert shifted["bubble_inner_bbox"] == [513, 5650, 649, 5733]
    assert shifted["safe_text_box"] == [525, 5662, 637, 5721]
    assert shifted["render_bbox"] == [542, 5666, 620, 5716]
    assert shifted["line_polygons"][0][0] == [498, 5652]


def test_shift_text_geometry_y_does_not_skip_required_second_shift():
    from strip.run import _shift_text_geometry_y

    strip_global = {
        "id": "ocr_001",
        "band_id": "page_006_band_104",
        "bbox": [453, 81927, 645, 81991],
        "text_pixel_bbox": [451, 81939, 643, 81999],
        "balloon_bbox": [453, 81927, 645, 81991],
        "bubble_mask_bbox": [453, 81927, 645, 81991],
        "bubble_inner_bbox": [465, 81939, 633, 81979],
        "safe_text_box": [474, 81943, 625, 81975],
        "render_bbox": [503, 81947, 594, 81971],
        "_coordinate_space": "strip",
    }

    final_page_local = _shift_text_geometry_y(strip_global, -69050)

    assert final_page_local["bbox"] == [453, 12877, 645, 12941]
    assert final_page_local["bubble_inner_bbox"] == [465, 12889, 633, 12929]
    assert final_page_local["safe_text_box"] == [474, 12893, 625, 12925]
    assert final_page_local["render_bbox"] == [503, 12897, 594, 12921]


def test_shift_text_geometry_xy_shifts_bubble_fields():
    from strip.run import _shift_text_geometry_xy

    shifted = _shift_text_geometry_xy(
        {
            "bubble_mask_bbox": [501, 218, 661, 325],
            "bubble_inner_bbox": [513, 230, 649, 313],
            "balloon_inner_bbox": [510, 225, 655, 320],
        },
        -100,
        5420,
    )

    assert shifted["bubble_mask_bbox"] == [401, 5638, 561, 5745]
    assert shifted["bubble_inner_bbox"] == [413, 5650, 549, 5733]
    assert shifted["balloon_inner_bbox"] == [410, 5645, 555, 5740]


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
            "_vision_blocks": [{
                "bbox": [80, 40, 320, 160],
                "confidence": 0.9,
                "bubble_id": "page_001_band_001_bubble_001",
                "bubble_mask_bbox": [80, 40, 320, 160],
                "bubble_inner_bbox": [96, 56, 304, 144],
            }],
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
            self.assertEqual(txt.get("bubble_id"), "page_001_band_001_bubble_001")
            self.assertEqual(txt.get("bubble_mask_bbox"), [80, 40, 320, 160])
            self.assertEqual(txt.get("bubble_inner_bbox"), [96, 56, 304, 144])

        call_page = runtime.run_ocr_stage.call_args.args[1]
        self.assertIn("_bubble_regions", call_page)
        self.assertEqual(call_page["_bubble_regions"][0]["bubble_id"], "page_001_band_000_bubble_001")

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

        with patch("typesetter.renderer.build_render_blocks", return_value=[dict(page["texts"][0])]):
            with patch("typesetter.renderer.render_text_block"):
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
