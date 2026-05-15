"""Regressões do adapter de inpaint do pipeline strip."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image


_FAST_WHITE_ENV = {
    "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
    "TRADUZAI_STRIP_FAST_WHITE_NARRATION": "1",
    "TRADUZAI_STRIP_FAST_WHITE_POST_CLEANUP": "0",
}
_FAST_LOCAL_ENV = {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}
_FAST_ALL_ENV = {**_FAST_WHITE_ENV, **_FAST_LOCAL_ENV}


class StripInpaintAdapterTests(unittest.TestCase):
    def test_fast_white_balloon_fill_is_opt_in_by_default(self):
        from inpainter import _fast_white_balloon_fill_enabled

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_fast_white_balloon_fill_enabled())

    def test_prewarm_band_inpainter_delegates_to_runtime_cache(self):
        import inpainter

        with patch("vision_stack.runtime._get_inpainter", return_value="warm") as get_inpainter:
            result = inpainter.prewarm_band_inpainter()

        self.assertEqual(result, "warm")
        get_inpainter.assert_called_once_with("quality")

    def test_delegates_to_runtime_inpainting_round(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 255, dtype=np.uint8)
        page = {
            "texts": [{"id": "t1", "bbox": [60, 20, 160, 60], "tipo": "fala"}],
            "_vision_blocks": [{"bbox": [54, 14, 166, 66], "confidence": 0.92}],
        }
        expected = np.full_like(band, 127)

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter") as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            return_value=expected,
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        self.assertEqual(cleaned[30, 80].tolist(), [127, 127, 127])
        self.assertEqual(cleaned[0, 0].tolist(), band[0, 0].tolist())
        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        args = apply_round.call_args[0]
        self.assertTrue(np.array_equal(args[0], band))
        self.assertEqual(args[1]["texts"], page["texts"])
        self.assertEqual(args[1]["_vision_blocks"], page["_vision_blocks"])
        self.assertEqual(args[2], "fake-inpainter")
        self.assertTrue(page["_strip_used_real_inpaint"])
        self.assertTrue(page["_strip_used_post_cleanup"])

    def test_fast_white_balloon_fill_skips_lama_when_all_blocks_are_covered(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 230, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [80, 35, 140, 55],
                    "balloon_bbox": [50, 20, 170, 75],
                    "tipo": "fala",
                    "text": "HELLO",
                }
            ],
            "_vision_blocks": [{"bbox": [76, 32, 144, 58], "confidence": 0.92}],
        }
        filled = band.copy()
        filled[25:70, 55:165] = 255

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=[50, 20, 170, 75]
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            return_value=filled,
        ) as apply_fill, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ) as post_cleanup, patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        apply_fill.assert_called_once()
        post_cleanup.assert_not_called()
        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertTrue(np.array_equal(cleaned, filled))
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_white_balloon_fill_includes_white_narration(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 230, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [80, 35, 140, 55],
                    "balloon_bbox": [50, 20, 170, 75],
                    "tipo": "narracao",
                    "balloon_type": "white",
                    "layout_profile": "top_narration",
                    "text": "BECAUSE.",
                }
            ],
            "_vision_blocks": [{"bbox": [76, 32, 144, 58], "confidence": 0.92}],
        }
        filled = band.copy()
        filled[25:70, 55:165] = 255

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=[50, 20, 170, 75]
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            return_value=filled,
        ) as apply_fill, patch("vision_stack.runtime._get_inpainter") as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        apply_fill.assert_called_once()
        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertTrue(np.array_equal(cleaned, filled))
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_white_balloon_fill_resolves_each_mask_from_original_band(self):
        from inpainter import inpaint_band_image

        band = np.full((120, 300, 3), 42, dtype=np.uint8)
        band[30:86, 40:120] = 245
        band[30:86, 180:260] = 245
        page = {
            "texts": [
                {
                    "id": "left",
                    "bbox": [58, 44, 102, 70],
                    "text_pixel_bbox": [58, 44, 102, 70],
                    "balloon_bbox": [32, 24, 128, 92],
                    "tipo": "fala",
                    "text": "LEFT",
                },
                {
                    "id": "right",
                    "bbox": [198, 44, 242, 70],
                    "text_pixel_bbox": [198, 44, 242, 70],
                    "balloon_bbox": [172, 24, 268, 92],
                    "tipo": "fala",
                    "text": "RIGHT",
                },
            ],
            "_vision_blocks": [
                {"bbox": [58, 44, 102, 70], "confidence": 0.94},
                {"bbox": [198, 44, 242, 70], "confidence": 0.94},
            ],
        }

        def _fake_fill(image_np, bbox):
            filled = image_np.copy()
            x1, y1, x2, y2 = bbox
            filled[y1:y2, x1:x2] = 255
            if not np.array_equal(image_np, band):
                filled[46:74, 135:165] = 255
            return filled

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox",
            side_effect=[[40, 30, 120, 86], [180, 30, 260, 86]],
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            side_effect=_fake_fill,
        ), patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertEqual(int(cleaned[60, 150, 0]), 42)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 2)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_white_balloon_fill_seeds_mask_from_text_bbox_not_overbroad_balloon_bbox(self):
        from inpainter import inpaint_band_image

        band = np.full((140, 320, 3), 36, dtype=np.uint8)
        band[38:102, 210:282] = 245
        page = {
            "texts": [
                {
                    "id": "right",
                    "bbox": [222, 52, 266, 82],
                    "text_pixel_bbox": [222, 52, 266, 82],
                    "balloon_bbox": [24, 0, 320, 140],
                    "tipo": "fala",
                    "text": "RIGHT",
                }
            ],
            "_vision_blocks": [{"bbox": [222, 52, 266, 82], "confidence": 0.94}],
        }

        def _fake_resolve(_image_np, payload):
            if payload.get("bbox") == [24, 0, 320, 140]:
                return [24, 0, 320, 140]
            return [206, 34, 286, 106]

        def _fake_fill(image_np, bbox):
            filled = image_np.copy()
            x1, y1, x2, y2 = bbox
            filled[y1:y2, x1:x2] = 255
            return filled

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox",
            side_effect=_fake_resolve,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            side_effect=_fake_fill,
        ), patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertEqual(int(cleaned[70, 120, 0]), 36)
        self.assertEqual(int(cleaned[70, 240, 0]), 255)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)

    def test_fast_white_debug_records_actual_fast_fill_changed_mask(self):
        from inpainter import inpaint_band_image

        band = np.full((80, 160, 3), 245, dtype=np.uint8)
        page = {
            "_band_index": 5,
            "_source_page_number": 2,
            "texts": [
                {
                    "id": "t1",
                    "bbox": [56, 28, 100, 48],
                    "text_pixel_bbox": [56, 28, 100, 48],
                    "balloon_bbox": [40, 16, 120, 64],
                    "tipo": "fala",
                    "text": "HELLO",
                    "balloon_type": "white",
                }
            ],
            "_vision_blocks": [{"bbox": [56, 28, 100, 48], "confidence": 0.92}],
        }
        filled = band.copy()
        filled[20:60, 44:116] = 255

        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                **_FAST_WHITE_ENV,
                "TRADUZAI_INPAINT_DEBUG_DIR": tmp,
            },
        ), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox",
            return_value=[40, 16, 120, 64],
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            return_value=filled,
        ), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=AssertionError("fast fill should consume covered block"),
        ):
            inpaint_band_image(band, page)
            debug_dir = Path(tmp) / "page_002_band_005"
            fast_mask = np.array(Image.open(debug_dir / "01_fast_fill_changed_mask.png").convert("L"))

        self.assertGreater(int(np.count_nonzero(fast_mask)), 0)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)

    def test_fast_white_narration_is_enabled_only_when_flag_is_on_for_speed_profile(self):
        from inpainter import _fast_white_rejection_reason

        text = {
            "tipo": "narracao",
            "balloon_type": "white",
            "layout_profile": "top_narration",
            "confidence": 0.91,
        }

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_fast_white_rejection_reason(text), "narration_disabled")
        with patch.dict(os.environ, {"TRADUZAI_STRIP_FAST_WHITE_NARRATION": "1"}, clear=True):
            self.assertEqual(_fast_white_rejection_reason(text), "")

    def test_fast_white_balloon_fill_accepts_edge_clipped_strip_balloon(self):
        from inpainter import inpaint_band_image

        band = np.full((224, 1200, 3), 255, dtype=np.uint8)
        cv2 = __import__("cv2")
        cv2.ellipse(band, (606, 112), (174, 96), 0, 0, 360, (0, 0, 0), 4)
        cv2.putText(
            band,
            "KAOR",
            (540, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [432, 16, 781, 208],
                    "text_pixel_bbox": [437, 19, 774, 205],
                    "balloon_bbox": [432, 16, 781, 208],
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "I AM THE CAPTAIN OF THE CERBERUS MERCENARIES, KAOR.",
                }
            ],
            "_vision_blocks": [{"bbox": [432, 16, 781, 208], "confidence": 0.95}],
        }

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=None
        ), patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertEqual(cleaned.shape, band.shape)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_white_balloon_fill_ignores_textured_narration(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 230, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [80, 35, 140, 55],
                    "balloon_bbox": [50, 20, 170, 75],
                    "tipo": "narracao",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "TITLE",
                }
            ],
            "_vision_blocks": [{"bbox": [76, 32, 144, 58], "confidence": 0.92}],
        }

        with patch.dict(os.environ, _FAST_ALL_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox"
        ) as resolve_white, patch(
            "vision_stack.runtime._apply_white_balloon_fill",
        ) as apply_fill, patch(
            "vision_stack.runtime._try_koharu_balloon_fill",
            return_value=None,
        ) as local_fill, patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter") as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        resolve_white.assert_not_called()
        apply_fill.assert_not_called()
        local_fill.assert_called_once()
        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        self.assertEqual(cleaned.shape, band.shape)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 0)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 1)

    def test_fast_local_balloon_fill_skips_lama_for_solid_textured_narration(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 300, 3), 245, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [90, 34, 180, 58],
                    "text_pixel_bbox": [94, 36, 176, 54],
                    "balloon_bbox": [70, 18, 210, 76],
                    "tipo": "narracao",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "AMAZING.",
                }
            ],
            "_vision_blocks": [{"bbox": [88, 32, 182, 60], "confidence": 0.88}],
        }
        filled = band.copy()
        filled[20:74, 72:208] = 246

        def _local_fill(image_np, mask):
            self.assertGreater(int(np.count_nonzero(mask[36:54, 94:176])), 0)
            return filled

        with patch.dict(os.environ, _FAST_LOCAL_ENV), patch(
            "vision_stack.runtime._try_koharu_balloon_fill", side_effect=_local_fill
        ) as local_fill, patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        local_fill.assert_called_once()
        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertTrue(np.array_equal(cleaned, filled))
        self.assertEqual(page["_strip_fast_white_balloon_count"], 0)
        self.assertEqual(page["_strip_fast_local_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_local_solid_background_fill_skips_lama_when_contour_fill_declines(self):
        from inpainter import inpaint_band_image

        band = np.full((120, 360, 3), 8, dtype=np.uint8)
        band[38:82, 80:280] = 250
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [78, 34, 282, 86],
                    "text_pixel_bbox": [80, 38, 280, 82],
                    "balloon_bbox": [64, 24, 296, 96],
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "CAPTION",
                }
            ],
            "_vision_blocks": [{"bbox": [78, 34, 282, 86], "confidence": 0.91}],
        }
        band[46:74, 72:80] = 250
        band[46:74, 280:288] = 250

        with patch.dict(os.environ, _FAST_LOCAL_ENV), patch(
            "vision_stack.runtime._try_koharu_balloon_fill", return_value=None
        ) as contour_fill, patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
        ) as apply_round:
            cleaned = inpaint_band_image(band, page)

        contour_fill.assert_called_once()
        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertLess(int(cleaned[55, 180, 0]), 32)
        self.assertLess(int(cleaned[55, 74, 0]), 32)
        self.assertLess(int(cleaned[55, 286, 0]), 32)
        self.assertEqual(int(cleaned[10, 10, 0]), 8)
        self.assertEqual(page["_strip_fast_local_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_local_solid_background_leaves_large_uniform_dark_caption_for_lama(self):
        from inpainter import inpaint_band_image

        band = np.full((308, 720, 3), 0, dtype=np.uint8)
        band[23:291, 97:620] = 245
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [92, 16, 623, 292],
                    "text_pixel_bbox": [97, 23, 620, 291],
                    "balloon_bbox": [92, 16, 623, 292],
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "LONG CAPTION",
                }
            ],
            "_vision_blocks": [{"bbox": [92, 16, 623, 292], "confidence": 0.91}],
        }

        with patch.dict(os.environ, _FAST_LOCAL_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=None
        ), patch(
            "vision_stack.runtime._try_koharu_balloon_fill",
            return_value=None,
        ), patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        self.assertEqual(int(cleaned[120, 300, 0]), 245)
        self.assertEqual(int(cleaned[10, 10, 0]), 0)
        self.assertEqual(page["_strip_fast_local_balloon_count"], 0)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 1)

    def test_fast_local_solid_background_leaves_large_noisy_dark_caption_for_lama(self):
        from inpainter import inpaint_band_image

        band = np.full((240, 720, 3), 6, dtype=np.uint8)
        band[42:198, 70:650] = 245
        band[12:36, 40:680] = [42, 48, 66]
        band[204:228, 40:680] = [3, 5, 28]
        band[36:204, 40:64] = [78, 26, 10]
        band[36:204, 656:680] = [0, 55, 86]
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [64, 36, 656, 204],
                    "text_pixel_bbox": [70, 42, 650, 198],
                    "balloon_bbox": [64, 36, 656, 204],
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "layout_profile": "standard",
                    "text": "LONG CAPTION",
                }
            ],
            "_vision_blocks": [{"bbox": [64, 36, 656, 204], "confidence": 0.91}],
        }

        with patch.dict(os.environ, _FAST_LOCAL_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=None
        ), patch(
            "vision_stack.runtime._try_koharu_balloon_fill",
            return_value=None,
        ), patch(
            "vision_stack.runtime._get_inpainter",
            return_value="fake-inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        self.assertEqual(cleaned.shape, band.shape)
        self.assertEqual(page["_strip_fast_local_balloon_count"], 0)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 1)

    def test_fast_metadata_background_fill_is_enabled_by_default_for_solid_light_ocr_box(self):
        from inpainter import inpaint_band_image

        band = np.full((160, 360, 3), 248, dtype=np.uint8)
        band[18:36, 24:336] = 32
        band[128:144, 24:336] = 42
        band[68:84, 118:244] = 12
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [92, 52, 270, 100],
                    "text_pixel_bbox": [118, 68, 244, 84],
                    "line_polygons": [[[118, 68], [244, 68], [244, 84], [118, 84]]],
                    "balloon_bbox": [72, 40, 292, 114],
                    "tipo": "narracao",
                    "balloon_type": "textured",
                    "layout_profile": "top_narration",
                    "background_rgb": [248, 248, 248],
                    "text": "READ ON",
                }
            ],
            "_vision_blocks": [{"bbox": [112, 62, 250, 90], "confidence": 0.76}],
        }

        env = {
            **_FAST_LOCAL_ENV,
            "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
        }
        with patch.dict(os.environ, env), patch(
            "vision_stack.runtime._try_koharu_balloon_fill",
            return_value=None,
        ), patch(
            "inpainter._try_solid_background_text_fill",
            return_value=None,
        ), patch(
            "vision_stack.runtime._get_inpainter",
        ) as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=lambda image_np, payload, inp: image_np.copy(),
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_not_called()
        apply_round.assert_not_called()
        self.assertTrue(np.all(cleaned[72:80, 128:234] == 248))
        self.assertEqual(int(cleaned[24, 64, 0]), 32)
        self.assertEqual(page["_strip_fast_local_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 0)

    def test_fast_white_balloon_fill_leaves_textured_blocks_for_lama(self):
        from inpainter import inpaint_band_image

        band = np.full((120, 320, 3), 210, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [70, 32, 132, 54],
                    "balloon_bbox": [48, 18, 166, 78],
                    "tipo": "fala",
                    "text": "HELLO",
                },
                {
                    "id": "t2",
                    "bbox": [220, 62, 286, 92],
                    "balloon_bbox": [204, 48, 300, 105],
                    "tipo": "sfx",
                    "text": "BANG",
                },
            ],
            "_vision_blocks": [
                {"bbox": [66, 28, 136, 58], "confidence": 0.92},
                {"bbox": [214, 58, 292, 96], "confidence": 0.88},
            ],
        }
        filled = band.copy()
        filled[18:78, 48:166] = 255

        def _capture_round(image_np, payload, inpainter):
            self.assertEqual(inpainter, "fake-inpainter")
            self.assertTrue(np.array_equal(image_np, filled))
            self.assertEqual(payload["_vision_blocks"], [{"bbox": [214, 58, 292, 96], "confidence": 0.88}])
            return image_np.copy()

        with patch.dict(os.environ, _FAST_WHITE_ENV), patch(
            "vision_stack.runtime._resolve_white_balloon_bbox", return_value=[48, 18, 166, 78]
        ), patch(
            "vision_stack.runtime._apply_white_balloon_fill",
            return_value=filled,
        ), patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter") as get_inpainter, patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=_capture_round,
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        get_inpainter.assert_called_once_with("quality")
        apply_round.assert_called_once()
        self.assertEqual(cleaned.shape, band.shape)
        self.assertEqual(page["_strip_fast_white_balloon_count"], 1)
        self.assertEqual(page["_strip_remaining_inpaint_blocks"], 1)

    def test_synthesizes_vision_blocks_from_texts_when_missing(self):
        from inpainter import inpaint_band_image

        band = np.full((80, 200, 3), 240, dtype=np.uint8)
        page = {
            "texts": [
                {
                    "id": "t1",
                    "bbox": [20, 12, 100, 42],
                    "text_pixel_bbox": [24, 16, 96, 38],
                    "tipo": "fala",
                    "text": "HELLO",
                }
            ]
        }

        def _capture_payload(image_np, payload, inpainter):
            self.assertEqual(inpainter, "fake-inpainter")
            self.assertEqual(len(payload["_vision_blocks"]), 1)
            self.assertEqual(payload["_vision_blocks"][0]["bbox"], [24, 16, 96, 38])
            return image_np.copy()

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=_capture_payload,
        ) as apply_round, patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        self.assertEqual(cleaned.shape, band.shape)
        apply_round.assert_called_once()

    def test_texts_without_geometry_are_skipped(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 255, dtype=np.uint8)
        page = {
            "texts": [
                {"id": "t_bad", "tipo": "sfx", "text": "..."},
                {"id": "t_ok", "bbox": [10, 10, 80, 40]},
            ]
        }

        def _capture_payload(image_np, payload, inpainter):
            self.assertEqual(len(payload["_vision_blocks"]), 1)
            self.assertEqual(payload["_vision_blocks"][0]["bbox"], [10, 10, 80, 40])
            return image_np.copy()

        with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
            "vision_stack.runtime._apply_inpainting_round",
            side_effect=_capture_payload,
        ), patch(
            "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
            side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
        ):
            cleaned = inpaint_band_image(band, page)

        self.assertEqual(cleaned.shape, band.shape)


class InpaintPassthroughTests(unittest.TestCase):
    def test_empty_texts_returns_copy(self):
        from inpainter import inpaint_band_image

        band = np.full((100, 200, 3), 128, dtype=np.uint8)
        result = inpaint_band_image(band, {"texts": []})
        self.assertEqual(result.shape, band.shape)
        self.assertTrue(np.array_equal(result, band))
        self.assertIsNot(result, band)

    def test_empty_image_returns_copy(self):
        from inpainter import inpaint_band_image

        band = np.zeros((0, 200, 3), dtype=np.uint8)
        page = {"texts": [{"id": "t", "bbox": [0, 0, 100, 50], "tipo": "fala"}]}
        result = inpaint_band_image(band, page)
        self.assertEqual(result.size, 0)

    def test_output_shape_preserved(self):
        from inpainter import inpaint_band_image

        for shape in [(50, 100, 3), (200, 400, 3), (1024, 800, 3)]:
            band = np.full(shape, 200, dtype=np.uint8)
            page = {"texts": [{"id": "t", "bbox": [10, 10, 50, 30], "tipo": "fala"}]}
            with patch("vision_stack.runtime._get_inpainter", return_value="fake-inpainter"), patch(
                "vision_stack.runtime._apply_inpainting_round",
                side_effect=lambda image_np, payload, inp: image_np.copy(),
            ), patch(
                "vision_stack.runtime._apply_post_inpaint_cleanup_timed",
                side_effect=lambda original, cleaned, texts, **_kwargs: (cleaned, {}),
            ):
                result = inpaint_band_image(band, page)
            self.assertEqual(result.shape, shape, f"Shape {shape} foi alterado")
