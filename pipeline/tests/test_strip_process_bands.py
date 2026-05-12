"""Testes de process_bands.py."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BandToPageDictTests(unittest.TestCase):
    def test_band_to_page_dict_remaps_balloon_coords_to_local(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=500,
            y_bottom=600,
            balloons=[
                Balloon(strip_bbox=BBox(50, 510, 150, 590), confidence=0.9),
            ],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0)

        self.assertEqual(page_dict["width"], 300)
        self.assertEqual(page_dict["height"], 100)
        self.assertEqual(page_dict["numero"], 1)
        self.assertEqual(page_dict["_band_index"], 1)
        block = page_dict["_vision_blocks"][0]
        self.assertEqual(block["bbox"], [50, 10, 150, 90])

    def test_band_to_page_dict_can_use_source_page_number_for_ocr_profile(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((120, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=900,
            y_bottom=1020,
            balloons=[Balloon(strip_bbox=BBox(40, 930, 180, 990), confidence=0.86)],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=12, source_page_number=2)

        self.assertEqual(page_dict["numero"], 2)
        self.assertEqual(page_dict["_source_page_number"], 2)
        self.assertEqual(page_dict["_band_index"], 13)


class CopyBackOutsideBalloonsTests(unittest.TestCase):
    def test_copy_back_preserves_pixels_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 200, dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        self.assertEqual(result[50, 100, 0], 200)
        self.assertEqual(result[5, 5, 0], 50)
        outside_y_top = result[:12, :, :]
        self.assertTrue(np.array_equal(outside_y_top, original[:12, :, :]))

    def test_copy_back_diff_below_2_outside_balloons(self):
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(42)
        original = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            original_slice=original,
            rendered_slice=rendered,
        )

        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        mask_inside = np.zeros(result.shape[:2], dtype=bool)
        mask_inside[12:88, 42:158] = True

        diff = np.abs(result.astype(np.int16) - original.astype(np.int16))
        # Fora da banda interna, pixels devem ser identicos ao original
        self.assertTrue(np.all(diff[~mask_inside] == 0),
            "Pixels fora do bbox+margin foram alterados pelo copy-back")

    def test_copy_back_pixel_perfect_outside_balloon(self):
        """Criterio Q2=a: pixels fora do balloon bbox+margin devem ser identicos ao original."""
        from strip.process_bands import _apply_copy_back_outside_balloons
        from strip.types import Band, Balloon, BBox
        import numpy as np

        rng = np.random.default_rng(7)
        original = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        rendered = rng.integers(0, 256, (300, 600, 3), dtype=np.uint8)
        band = Band(
            y_top=0, y_bottom=300,
            balloons=[Balloon(strip_bbox=BBox(100, 50, 300, 200), confidence=0.9)],
            original_slice=original.copy(),
            rendered_slice=rendered.copy(),
        )
        result = _apply_copy_back_outside_balloons(band, balloon_margin=8)

        # Fora do bbox+margin (y < 42, x < 92, etc.) deve ser identico ao original
        self.assertTrue(np.array_equal(result[:42, :, :], original[:42, :, :]),
            "Rows acima do balloon nao sao pixel-perfect iguais ao original")
        # Dentro do bbox deve ser identico ao rendered
        self.assertTrue(np.array_equal(result[60:190, 110:290, :], rendered[60:190, 110:290, :]),
            "Interior do balloon nao e identico ao rendered")


class SmartSkipShadowTests(unittest.TestCase):
    def test_apply_smart_skip_shadow_records_audit_without_skip_processing_mutation(self):
        from strip.process_bands import _apply_smart_skip_shadow

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.0,
                    "bbox": [10, 10, 180, 40],
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.95,
                    "bbox": [20, 60, 220, 130],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        _apply_smart_skip_shadow(page, perf)

        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][1]["skip_processing"])
        self.assertEqual(page["_smart_skip_shadow"]["candidate_count"], 1)
        self.assertEqual(perf["smart_skip_shadow_candidate_count"], 1)
        self.assertEqual(perf["smart_skip_shadow_not_safe_count"], 1)
        self.assertEqual(
            perf["smart_skip_shadow_category_counts"]["credit_or_watermark"],
            1,
        )

    def test_apply_smart_skip_real_marks_only_all_safe_bands(self):
        from strip.process_bands import _apply_smart_skip_real

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "All comics on this website are just previews...",
                    "confidence": 0.95,
                    "bbox": [10, 10, 220, 60],
                    "skip_processing": False,
                },
                {
                    "id": "timer",
                    "text": "00:00:05",
                    "confidence": 0.8,
                    "bbox": [20, 80, 120, 110],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        applied = _apply_smart_skip_real(page, perf)

        self.assertTrue(applied)
        self.assertTrue(page["texts"][0]["skip_processing"])
        self.assertTrue(page["texts"][1]["skip_processing"])
        self.assertEqual(page["texts"][0]["skip_reason"], "smart_skip")
        self.assertEqual(page["texts"][1]["skip_reason"], "smart_skip")
        self.assertIn("smart_skip_decision", page["texts"][0])
        self.assertEqual(perf["smart_skip_real_candidate_count"], 2)
        self.assertEqual(perf["smart_skip_real_not_safe_count"], 0)
        self.assertTrue(perf["smart_skip_real_applied"])

    def test_apply_smart_skip_real_does_not_mutate_mixed_bands(self):
        from strip.process_bands import _apply_smart_skip_real

        page = {
            "numero": 1,
            "texts": [
                {
                    "id": "credit",
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.95,
                    "bbox": [10, 10, 220, 60],
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.95,
                    "bbox": [20, 80, 220, 130],
                    "skip_processing": False,
                },
            ],
        }
        perf = {}

        applied = _apply_smart_skip_real(page, perf)

        self.assertFalse(applied)
        self.assertFalse(page["texts"][0]["skip_processing"])
        self.assertFalse(page["texts"][1]["skip_processing"])
        self.assertNotIn("skip_reason", page["texts"][0])
        self.assertEqual(perf["smart_skip_real_candidate_count"], 1)
        self.assertEqual(perf["smart_skip_real_not_safe_count"], 1)
        self.assertFalse(perf["smart_skip_real_applied"])


class ProcessBandTests(unittest.TestCase):
    def _make_band(self):
        from strip.types import Band, Balloon, BBox
        import numpy as np
        slice_img = np.full((100, 300, 3), 200, dtype=np.uint8)
        return Band(
            y_top=0, y_bottom=100,
            balloons=[Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9)],
            strip_slice=slice_img.copy(),
            original_slice=slice_img.copy(),
        )

    def test_process_band_populates_rendered_slice(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        # Stages mockadas — só precisam retornar dict válido / ndarray
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
        ]}
        translator = MagicMock()
        translator.translate_pages.return_value = ([{
            "texts": [{"id": "t1", "translated": "OLÁ", "tipo": "fala", "bbox": [50, 20, 150, 80]}]
        }], [])
        inpainter = MagicMock()

        def fake_inpaint(_image, page):
            page["_strip_fast_white_balloon_count"] = 1
            page["_strip_fast_local_balloon_count"] = 2
            page["_strip_remaining_inpaint_blocks"] = 3
            page["_strip_fast_white_rejection_reasons"] = {"no_white_fill_mask": 4}
            page["_strip_fast_local_rejection_reasons"] = {"no_flat_fill": 5}
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        self.assertIsNotNone(band.rendered_slice)
        # Stages foram chamadas
        runtime.run_ocr_stage.assert_called_once()
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_called_once()
        typesetter.render_band_image.assert_called_once()
        self.assertIn("durations_sec", band.perf)
        self.assertIn("ocr", band.perf["durations_sec"])
        self.assertIn("translate", band.perf["durations_sec"])
        self.assertIn("inpaint", band.perf["durations_sec"])
        self.assertIn("typeset", band.perf["durations_sec"])
        self.assertEqual(band.ocr_result.get("_perf", {}).get("ocr_text_count"), 1)
        self.assertEqual(band.perf["fast_white_balloon_count"], 1)
        self.assertEqual(band.perf["fast_local_balloon_count"], 2)
        self.assertEqual(band.perf["remaining_inpaint_blocks"], 3)
        self.assertEqual(band.perf["fast_white_rejection_reasons"], {"no_white_fill_mask": 4})
        self.assertEqual(band.perf["fast_local_rejection_reasons"], {"no_flat_fill": 5})

    def test_process_band_notifies_ordered_context_after_translate_before_inpaint(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [
                {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}],
                "_glossary_additions": {"FENRIS": "Fenris"},
            }
        ]
        events = []

        def on_language_ready(page):
            events.append(("callback", page["texts"][0]["translated"], dict(page.get("_glossary_additions") or {})))
            page["texts"][0]["translated"] = "MUTATED"

        def fake_inpaint(_image, page):
            events.append(("inpaint", page["texts"][0]["translated"]))
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            ordered_context_after_translate_callback=on_language_ready,
        )

        self.assertEqual(events[0], ("callback", "OLA", {"FENRIS": "Fenris"}))
        self.assertEqual(events[1], ("inpaint", "OLA"))

    def test_process_band_serializes_gpu_stages_when_lock_is_provided(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        class TrackingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("lock_enter")

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("lock_exit")

        band = self._make_band()
        runtime = MagicMock()

        def fake_ocr(_image, _page):
            lock.events.append("ocr")
            return {
                "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
                "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            }

        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]

        def fake_inpaint(_image, page):
            lock.events.append("inpaint")
            return np.full((100, 300, 3), 255, dtype=np.uint8)

        lock = TrackingLock()
        runtime.run_ocr_stage.side_effect = fake_ocr
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = fake_inpaint
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            gpu_stage_lock=lock,
        )

        self.assertEqual(
            lock.events,
            ["lock_enter", "ocr", "lock_exit", "lock_enter", "inpaint", "lock_exit"],
        )

    def test_process_band_serializes_typeset_when_lock_is_provided(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        class TrackingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("typeset_lock_enter")

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("typeset_lock_exit")

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)

        lock = TrackingLock()

        def fake_typeset(_image, _page):
            lock.events.append("typeset")
            return np.full((100, 300, 3), 100, dtype=np.uint8)

        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = fake_typeset

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            typeset_stage_lock=lock,
        )

        self.assertEqual(lock.events, ["typeset_lock_enter", "typeset", "typeset_lock_exit"])

    def test_process_band_restores_ocr_metadata_when_translation_payload_is_reduced(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text_pixel_bbox": [62, 28, 140, 72],
                    "line_polygons": [[[62, 28], [140, 28], [140, 72], [62, 72]]],
                    "text": "HELLO",
                    "tipo": "fala",
                    "ocr_source": "paddleocr",
                    "ocr_confidence": 0.91,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}],
                "_vision_blocks": [],
            }
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        inpaint_page = inpainter.inpaint_band_image.call_args[0][1]
        self.assertEqual(inpaint_page["texts"][0]["text_pixel_bbox"], [62, 28, 140, 72])
        self.assertEqual(inpaint_page["texts"][0]["ocr_source"], "paddleocr")
        self.assertEqual(inpaint_page["texts"][0]["bbox"], [50, 20, 150, 80])
        self.assertEqual(inpaint_page["_vision_blocks"][0]["bbox"], [40, 10, 160, 90])

        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")
        self.assertEqual(band.ocr_result["texts"][0]["line_polygons"][0][0], [62, 28])

    def test_process_band_forwards_translation_runtime_options(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "text": "HELLO", "tipo": "fala"},
        ]}
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _: img.copy()
        translation_context = {"memory": [{"source": "HELLO", "target": "OLA"}]}

        process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
            models_dir="D:/traduzai_data/models",
            ollama_host="http://127.0.0.1:11435",
            ollama_model="custom-translator",
            translation_context=translation_context,
        )

        kwargs = translator.translate_pages.call_args.kwargs
        self.assertEqual(kwargs["models_dir"], "D:/traduzai_data/models")
        self.assertEqual(kwargs["ollama_host"], "http://127.0.0.1:11435")
        self.assertEqual(kwargs["ollama_model"], "custom-translator")
        self.assertIs(kwargs["translation_context"], translation_context)

    def test_process_band_applies_smart_skip_shadow_only_when_flag_is_enabled(self):
        from unittest.mock import MagicMock, patch
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "credit",
                    "bbox": [50, 20, 150, 40],
                    "text": "FOR FASTER UPDATE",
                    "confidence": 0.0,
                    "tipo": "fala",
                    "skip_processing": False,
                },
                {
                    "id": "dialogue",
                    "bbox": [50, 50, 160, 80],
                    "text": "IS THIS RECORDING?",
                    "confidence": 0.9,
                    "tipo": "fala",
                    "skip_processing": False,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 170, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.side_effect = lambda pages, **_kw: pages
        inpainter = MagicMock()
        inpainter.inpaint_band_image.side_effect = lambda img, _page: img.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.side_effect = lambda img, _page: img.copy()

        with patch.dict("os.environ", {"TRADUZAI_SMART_SKIP_SHADOW": "1"}):
            process_band(
                band,
                runtime=runtime,
                translator=translator,
                inpainter=inpainter,
                typesetter=typesetter,
                page_idx=0,
            )

        translated_input = translator.translate_pages.call_args.args[0][0]
        self.assertEqual(translated_input["_smart_skip_shadow"]["candidate_count"], 1)
        self.assertFalse(translated_input["texts"][0]["skip_processing"])
        self.assertFalse(translated_input["texts"][1]["skip_processing"])
        self.assertEqual(band.perf["smart_skip_shadow_candidate_count"], 1)
        self.assertEqual(band.perf["smart_skip_shadow_not_safe_count"], 1)

    def test_process_band_passes_source_page_number_to_ocr_runtime(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {"texts": []}

        process_band(
            band,
            runtime=runtime,
            translator=MagicMock(),
            inpainter=MagicMock(),
            typesetter=MagicMock(),
            page_idx=9,
            source_page_number=2,
        )

        page_dict = runtime.run_ocr_stage.call_args.args[1]
        self.assertEqual(page_dict["numero"], 2)
        self.assertEqual(page_dict["_source_page_number"], 2)
        self.assertEqual(page_dict["_band_index"], 10)

    def test_process_band_uses_precomputed_ocr_page_without_runtime_call(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = AssertionError("runtime OCR should be skipped")
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "HELLO",
                    "tipo": "fala",
                    "confidence": 0.93,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
            "_ocr_stats": {
                "full_page_mapped": 1,
                "macro_ocr_real": True,
                "macro_window_count": 1,
                "macro_ocr_block_count": 1,
                "macro_ocr_empty_record_count": 0,
            },
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA", "tipo": "fala"}]}
        ]
        inpainter = MagicMock()
        inpainter.inpaint_band_image.return_value = np.full((100, 300, 3), 255, dtype=np.uint8)
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = np.full((100, 300, 3), 100, dtype=np.uint8)

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=4,
            source_page_number=2,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        self.assertIs(result, band)
        runtime.run_ocr_stage.assert_not_called()
        translator.translate_pages.assert_called_once()
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["numero"], 2)
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["width"], 300)
        self.assertEqual(translator.translate_pages.call_args.args[0][0]["height"], 100)
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")
        self.assertTrue(band.perf["ocr_precomputed_page"])
        self.assertTrue(band.ocr_result["_perf"]["ocr_precomputed_page"])
        self.assertEqual(band.perf["ocr_full_page_mapped"], 1)
        self.assertTrue(band.perf["ocr_macro_ocr_real"])
        self.assertEqual(band.perf["ocr_macro_window_count"], 1)
        self.assertEqual(band.perf["ocr_macro_ocr_block_count"], 1)
        self.assertEqual(band.perf["ocr_macro_ocr_empty_record_count"], 0)

    def test_ocr_stage_result_is_snapshot_and_skips_runtime_for_precomputed_page(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        band = self._make_band()
        page_dict = process_bands._band_to_page_dict(band, page_idx=3, source_page_number=2)
        precomputed_ocr_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        runtime = MagicMock()
        runtime.run_ocr_stage.side_effect = AssertionError("runtime OCR should be skipped")

        output = process_bands._run_band_ocr_stage(
            band,
            runtime=runtime,
            page_dict=page_dict,
            precomputed_ocr_page=precomputed_ocr_page,
        )

        runtime.run_ocr_stage.assert_not_called()
        self.assertEqual(output.stage_id, "ocr")
        self.assertEqual(dict(output.perf_updates), {
            "ocr_precomputed_page": True,
            "ocr_runtime_skipped": True,
        })

        page_snapshot = output.to_page_dict()
        self.assertEqual(page_snapshot["numero"], 2)
        self.assertEqual(page_snapshot["_band_index"], 4)
        page_snapshot["texts"][0]["text"] = "MUTATED"
        precomputed_ocr_page["texts"][0]["text"] = "SOURCE MUTATED"

        self.assertEqual(output.to_page_dict()["texts"][0]["text"], "HELLO")

    def test_translate_stage_result_merges_ocr_metadata_as_snapshot(self):
        from unittest.mock import MagicMock
        from strip import process_bands

        ocr_page = {
            "numero": 2,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "line_polygons": [[[60, 30], [140, 30], [140, 70], [60, 70]]],
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {"texts": [{"id": "t1", "translated": "OLA"}], "_vision_blocks": []}
        ]

        output = process_bands._run_translate_stage(
            ocr_page,
            translator=translator,
            context={"obra": "Demo"},
            glossario={"HELLO": "OLA"},
            idioma_origem="en",
            idioma_destino="pt-BR",
            obra="Demo",
            models_dir="D:/models",
            ollama_host="http://127.0.0.1:11434",
            ollama_model="model",
            translation_context={"memory": []},
        )

        self.assertEqual(output.stage_id, "translate")
        translated = output.to_page_dict()
        self.assertEqual(translated["texts"][0]["translated"], "OLA")
        self.assertEqual(translated["texts"][0]["bbox"], [50, 20, 150, 80])
        self.assertEqual(translated["texts"][0]["line_polygons"][0][0], [60, 30])
        self.assertEqual(translated["_vision_blocks"][0]["bbox"], [40, 10, 160, 90])

        translated["texts"][0]["translated"] = "MUTATED"
        ocr_page["texts"][0]["bbox"] = [0, 0, 1, 1]
        self.assertEqual(output.to_page_dict()["texts"][0]["translated"], "OLA")
        self.assertEqual(output.to_page_dict()["texts"][0]["bbox"], [50, 20, 150, 80])

    def test_review_layout_stage_result_adds_balloon_bbox_as_snapshot(self):
        from strip import process_bands

        band = self._make_band()
        ocr_page = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "text": "HELLO",
                    "tipo": "fala",
                }
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }

        output = process_bands._run_review_layout_stage(
            band,
            ocr_page=ocr_page,
            band_history=[],
            connected_reasoner_config={"enabled": True},
        )

        self.assertEqual(output.stage_id, "review_layout")
        reviewed = output.to_page_dict()
        self.assertEqual(reviewed["texts"][0]["balloon_bbox"], [50, 20, 150, 80])
        self.assertEqual(reviewed["_connected_balloon_reasoner"], {"enabled": True})

        reviewed["texts"][0]["balloon_bbox"] = [0, 0, 1, 1]
        ocr_page["texts"][0]["bbox"] = [0, 0, 1, 1]
        self.assertEqual(output.to_page_dict()["texts"][0]["balloon_bbox"], [50, 20, 150, 80])

    def test_inpaint_stage_result_snapshots_image_and_perf_updates(self):
        from unittest.mock import MagicMock
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        translated_page = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "translated": "OLA"}],
        }
        inpainter = MagicMock()

        def fake_inpaint(_image, page):
            page["_strip_fast_white_balloon_count"] = 2
            page["_strip_fast_local_balloon_count"] = 1
            page["_strip_remaining_inpaint_blocks"] = 3
            page["_strip_used_real_inpaint"] = True
            page["_strip_fast_white_rejection_reasons"] = {"no_white_fill_mask": 4}
            return np.full((100, 300, 3), 210, dtype=np.uint8)

        inpainter.inpaint_band_image.side_effect = fake_inpaint

        output = process_bands._run_inpaint_stage(
            band,
            inpainter=inpainter,
            translated_page=translated_page,
        )

        self.assertEqual(output.stage_id, "inpaint")
        self.assertEqual(
            dict(output.perf_updates),
            {
                "fast_white_balloon_count": 2,
                "fast_local_balloon_count": 1,
                "remaining_inpaint_blocks": 3,
                "fast_white_rejection_reasons": {"no_white_fill_mask": 4},
                "used_real_inpaint": True,
            },
        )
        image_snapshot = output.to_image()
        image_snapshot[:, :, :] = 0
        self.assertEqual(int(output.to_image()[0, 0, 0]), 210)

    def test_typeset_and_copy_back_stage_results_snapshot_images_without_mutating_band(self):
        from unittest.mock import MagicMock
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        original = np.full((100, 300, 3), 50, dtype=np.uint8)
        cleaned = np.full((100, 300, 3), 180, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 220, dtype=np.uint8)
        band.original_slice = original.copy()
        band.rendered_slice = None
        translated_page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [50, 20, 150, 80],
                    "translated": "OLA",
                }
            ],
        }
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rendered.copy()

        typeset_output = process_bands._run_typeset_stage(
            cleaned,
            typesetter=typesetter,
            translated_page=translated_page,
        )
        copy_back_output = process_bands._run_copy_back_stage(
            band,
            rendered_slice=typeset_output.to_image(),
            translated_page=translated_page,
        )

        self.assertEqual(typeset_output.stage_id, "typeset")
        self.assertEqual(copy_back_output.stage_id, "copy_back")
        self.assertIsNone(band.rendered_slice)
        copy_back_image = copy_back_output.to_image()
        self.assertEqual(int(copy_back_image[50, 100, 0]), 220)
        self.assertEqual(int(copy_back_image[5, 5, 0]), 50)
        copy_back_image[:, :, :] = 0
        self.assertEqual(int(copy_back_output.to_image()[50, 100, 0]), 220)

    def test_commit_band_outputs_snapshots_final_band_state(self):
        import numpy as np
        from strip import process_bands

        band = self._make_band()
        cleaned = np.full((100, 300, 3), 180, dtype=np.uint8)
        rendered = np.full((100, 300, 3), 220, dtype=np.uint8)
        ocr_result = {
            "texts": [{"id": "t1", "bbox": [50, 20, 150, 80], "translated": "OLA"}],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90]}],
        }

        committed = process_bands._commit_band_outputs(
            band,
            cleaned_slice=cleaned,
            rendered_slice=rendered,
            ocr_result=ocr_result,
        )

        self.assertIs(committed, band)
        cleaned[:, :, :] = 0
        rendered[:, :, :] = 0
        ocr_result["texts"][0]["translated"] = "MUTATED"

        self.assertEqual(int(band.cleaned_slice[0, 0, 0]), 180)
        self.assertEqual(int(band.rendered_slice[0, 0, 0]), 220)
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "OLA")

    def test_process_band_with_no_balloons_returns_original_slice(self):
        from strip.process_bands import process_band
        from strip.types import Band
        import numpy as np
        from unittest.mock import MagicMock

        slice_img = np.full((50, 300, 3), 80, dtype=np.uint8)
        band = Band(y_top=0, y_bottom=50, balloons=[], strip_slice=slice_img.copy(), original_slice=slice_img.copy())

        result = process_band(
            band,
            runtime=MagicMock(),
            translator=MagicMock(),
            inpainter=MagicMock(),
            typesetter=MagicMock(),
            page_idx=0,
        )

        self.assertIs(result, band)

    def test_process_band_with_no_accepted_texts_skips_expensive_stages(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [],
            "_vision_blocks": [{"bbox": [50, 20, 150, 80], "confidence": 0.9}],
            "_ocr_stats": {
                "scanlation_credit_skipped": True,
                "cover_editorial_skipped": True,
                "block_count": 1,
                "full_page_mapped": 0,
                "crop_fallback_attempts": 0,
                "crop_fallback_recovered": 0,
            },
        }
        translator = MagicMock()
        inpainter = MagicMock()
        typesetter = MagicMock()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        runtime.run_ocr_stage.assert_called_once()
        translator.translate_pages.assert_not_called()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertEqual(band.ocr_result["texts"], [])
        self.assertEqual(band.ocr_result["_vision_blocks"], [])
        self.assertTrue(band.perf["ocr_scanlation_credit_skipped"])
        self.assertTrue(band.perf["ocr_cover_editorial_skipped"])

    def test_process_band_skips_repaint_when_all_texts_are_skip_processing(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "YOU...!!",
                    "original": "YOU...!!",
                    "translated": "YOU...!!",
                    "tipo": "narracao",
                    "skip_processing": True,
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        inpainter = MagicMock()
        typesetter = MagicMock()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        translator.translate_pages.assert_not_called()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertTrue(band.perf["skip_processing_copy"])
        self.assertTrue(band.ocr_result["texts"][0]["skip_processing"])

    def test_process_band_skips_repaint_when_translation_marks_all_texts_skip_processing(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "YOU...!!",
                    "original": "YOU...!!",
                    "tipo": "narracao",
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [
                    {
                        "id": "t1",
                        "original": "YOU...!!",
                        "translated": "YOU...!!",
                        "tipo": "narracao",
                        "skip_processing": True,
                    }
                ],
            }
        ]
        inpainter = MagicMock()
        typesetter = MagicMock()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertTrue(band.perf["skip_processing_copy"])
        self.assertTrue(band.ocr_result["texts"][0]["skip_processing"])

    def test_process_band_skips_repaint_when_all_translations_are_unchanged(self):
        from unittest.mock import MagicMock
        from strip.process_bands import process_band
        import numpy as np

        band = self._make_band()
        runtime = MagicMock()
        runtime.run_ocr_stage.return_value = {
            "numero": 1,
            "width": 300,
            "height": 100,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [50, 20, 150, 80],
                    "balloon_bbox": [40, 10, 160, 90],
                    "text": "HYAAH!!",
                    "original": "HYAAH!!",
                    "tipo": "narracao",
                },
            ],
            "_vision_blocks": [{"bbox": [40, 10, 160, 90], "confidence": 0.9}],
        }
        translator = MagicMock()
        translator.translate_pages.return_value = [
            {
                "texts": [
                    {
                        "id": "t1",
                        "translated": "HYAAH!!",
                        "original": "HYAAH!!",
                        "tipo": "narracao",
                    }
                ],
            }
        ]
        inpainter = MagicMock()
        typesetter = MagicMock()

        result = process_band(
            band,
            runtime=runtime,
            translator=translator,
            inpainter=inpainter,
            typesetter=typesetter,
            page_idx=0,
        )

        self.assertIs(result, band)
        translator.translate_pages.assert_called_once()
        inpainter.inpaint_band_image.assert_not_called()
        typesetter.render_band_image.assert_not_called()
        self.assertTrue(np.array_equal(band.cleaned_slice, band.original_slice))
        self.assertTrue(np.array_equal(band.rendered_slice, band.original_slice))
        self.assertTrue(band.perf["unchanged_translation_skip"])
        self.assertEqual(band.ocr_result["texts"][0]["translated"], "HYAAH!!")

class BandAdaptersTests(unittest.TestCase):
    def test_inpaint_band_image_returns_same_shape(self):
        from inpainter import inpaint_band_image
        import numpy as np
        band = np.full((100, 300, 3), 200, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala", "original": "HELLO"},
        ]}
        cleaned = inpaint_band_image(band, page)
        self.assertEqual(cleaned.shape, band.shape)

    def test_render_band_image_returns_same_shape(self):
        from typesetter.renderer import render_band_image
        import numpy as np
        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {"texts": [
            {"id": "t1", "bbox": [50, 20, 150, 80], "tipo": "fala",
             "balloon_bbox": [50, 20, 150, 80],
             "translated": "OLÁ",
             "estilo": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 24, "cor": "#000000"}},
        ]}
        rendered = render_band_image(band, page)
        self.assertEqual(rendered.shape, band.shape)

