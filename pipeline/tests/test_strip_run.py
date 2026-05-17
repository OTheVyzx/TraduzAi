"""Smoke test do entry-point run_chapter."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    def test_strip_band_margin_is_larger_for_cjk_sources(self):
        from strip.run import _strip_band_margin_px

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_strip_band_margin_px("ko"), 96)
            self.assertEqual(_strip_band_margin_px("ja"), 96)
            self.assertEqual(_strip_band_margin_px("zh-cn"), 96)
            self.assertEqual(_strip_band_margin_px("en"), 16)

    def test_strip_band_margin_env_override_wins(self):
        from strip.run import _strip_band_margin_px

        with patch.dict("os.environ", {"TRADUZAI_STRIP_BAND_MARGIN_PX": "48"}, clear=False):
            self.assertEqual(_strip_band_margin_px("ko"), 48)

    def test_page_cleanup_limit_mask_follows_text_geometry(self):
        from strip.run import _build_page_cleanup_limit_mask

        image = np.full((120, 180, 3), 255, dtype=np.uint8)
        text = {
            "bbox": [50, 40, 110, 64],
            "text_pixel_bbox": [56, 44, 104, 60],
            "line_polygons": [[[56, 44], [104, 44], [104, 60], [56, 60]]],
            "balloon_bbox": [20, 20, 150, 92],
            "balloon_type": "white",
        }

        mask = _build_page_cleanup_limit_mask(image, [text])

        self.assertIsNotNone(mask)
        self.assertGreater(int(mask[50, 80]), 0)
        self.assertEqual(int(mask[88, 80]), 0)

    def test_page_final_cleanup_skips_textured_translucent_balloons(self):
        from strip.run import _cleanup_page_inpaint_and_rerender

        original = np.full((120, 180, 3), 220, dtype=np.uint8)
        clean = original.copy()
        rendered = original.copy()
        texts = [
            {
                "bbox": [40, 40, 140, 70],
                "text_pixel_bbox": [40, 40, 140, 70],
                "translated": "TEXTO",
                "original": "TEXT",
                "balloon_type": "textured",
                "block_profile": "standard",
            }
        ]
        typesetter = MagicMock()

        with patch(
            "vision_stack.runtime._apply_white_balloon_near_text_residual_cleanup",
            side_effect=AssertionError("textured balloon should not run white cleanup"),
        ):
            fixed_clean, fixed_rendered, did_fix = _cleanup_page_inpaint_and_rerender(
                original_image=original,
                clean_image=clean,
                page_texts=texts,
                rendered_image=rendered,
                typesetter=typesetter,
            )

        self.assertFalse(did_fix)
        self.assertIs(fixed_clean, clean)
        self.assertIs(fixed_rendered, rendered)
        typesetter.render_band_image.assert_not_called()

    def test_page_final_near_text_cleanup_is_disabled_by_default(self):
        from strip.run import _cleanup_page_inpaint_and_rerender

        original = np.full((120, 180, 3), 255, dtype=np.uint8)
        clean = original.copy()
        rendered = original.copy()
        texts = [
            {
                "bbox": [40, 40, 140, 70],
                "text_pixel_bbox": [44, 44, 136, 66],
                "translated": "TEXTO",
                "original": "TEXT",
                "balloon_type": "white",
            }
        ]
        fixed = clean.copy()
        fixed[52, 80] = [244, 244, 244]
        rerendered = rendered.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rerendered

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_ENABLE_PAGE_FINAL_NEAR_TEXT_CLEANUP": "0",
                "TRADUZAI_PAGE_FINAL_FULL_CLEANUP": "0",
            },
            clear=False,
        ), patch(
            "vision_stack.runtime._white_cleanup_texts",
            return_value=texts,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_near_text_residual_cleanup",
            return_value=fixed,
        ) as cleanup:
            fixed_clean, fixed_rendered, did_fix = _cleanup_page_inpaint_and_rerender(
                original_image=original,
                clean_image=clean,
                page_texts=texts,
                rendered_image=rendered,
                typesetter=typesetter,
            )

        self.assertFalse(did_fix)
        self.assertIs(fixed_clean, clean)
        self.assertIs(fixed_rendered, rendered)
        cleanup.assert_not_called()
        typesetter.render_band_image.assert_not_called()

    def test_page_final_near_text_cleanup_can_be_enabled(self):
        from strip.run import _cleanup_page_inpaint_and_rerender

        original = np.full((120, 180, 3), 255, dtype=np.uint8)
        clean = original.copy()
        rendered = original.copy()
        texts = [
            {
                "bbox": [40, 40, 140, 70],
                "text_pixel_bbox": [44, 44, 136, 66],
                "translated": "TEXTO",
                "original": "TEXT",
                "balloon_type": "white",
            }
        ]
        fixed = clean.copy()
        fixed[52, 80] = [244, 244, 244]
        rerendered = rendered.copy()
        typesetter = MagicMock()
        typesetter.render_band_image.return_value = rerendered

        with patch.dict("os.environ", {"TRADUZAI_ENABLE_PAGE_FINAL_NEAR_TEXT_CLEANUP": "1"}, clear=True), patch(
            "vision_stack.runtime._white_cleanup_texts",
            return_value=texts,
        ), patch(
            "vision_stack.runtime._apply_white_balloon_near_text_residual_cleanup",
            return_value=fixed,
        ):
            fixed_clean, fixed_rendered, did_fix = _cleanup_page_inpaint_and_rerender(
                original_image=original,
                clean_image=clean,
                page_texts=texts,
                rendered_image=rendered,
                typesetter=typesetter,
            )

        self.assertTrue(did_fix)
        self.assertIs(fixed_clean, fixed)
        self.assertIs(fixed_rendered, rerendered)
        typesetter.render_band_image.assert_called_once()

    def test_inpaint_block_from_vision_block_preserves_text_geometry_without_connected_metadata(self):
        from strip.run import _inpaint_block_from_vision_block

        block = _inpaint_block_from_vision_block(
            {
                "bbox": [10, 20, 90, 70],
                "confidence": 0.91,
                "text_pixel_bbox": [18, 26, 82, 62],
                "line_polygons": [[[18, 26], [82, 26], [82, 62], [18, 62]]],
                "balloon_type": "white",
                "connected_lobe_bboxes": [[0, 0, 50, 70]],
                "balloon_subregions": [[0, 0, 50, 70]],
            }
        )

        self.assertEqual(block["bbox"], [10, 20, 90, 70])
        self.assertEqual(block["text_pixel_bbox"], [18, 26, 82, 62])
        self.assertEqual(block["line_polygons"][0][2], [82, 62])
        self.assertEqual(block["balloon_type"], "white")
        self.assertNotIn("connected_lobe_bboxes", block)
        self.assertNotIn("balloon_subregions", block)

    def test_finalize_output_page_ocr_metadata_runs_same_cleanup_after_band_assignment(self):
        from strip.run import _finalize_output_page_ocr_metadata
        from strip.types import OutputPage

        texts = [
            {
                "text": "PLEASE, FOR THE CHILD'S",
                "translated": "POR FAVOR, PELO BEM",
                "bbox": [498, 1655, 656, 1707],
                "source_bbox": [25, 1436, 667, 1708],
                "text_pixel_bbox": [498, 1655, 656, 1707],
                "line_polygons": [
                    [[498, 1652], [658, 1654], [658, 1678], [498, 1676]],
                    [[507, 1686], [648, 1686], [648, 1707], [507, 1707]],
                ],
                "confidence": 0.93,
                "tipo": "fala",
                "balloon_type": "textured",
            },
            {
                "text": "SAKE.",
                "translated": "DA CRIANCA.",
                "bbox": [546, 1720, 612, 1740],
                "source_bbox": [497, 1648, 660, 1741],
                "text_pixel_bbox": [546, 1720, 612, 1740],
                "line_polygons": [[[543, 1718], [615, 1718], [615, 1743], [543, 1743]]],
                "confidence": 0.94,
                "tipo": "fala",
                "balloon_type": "white",
                "block_profile": "white_balloon",
            },
            {
                "text": "THE SA",
                "translated": "O SA",
                "bbox": [320, 1699, 562, 1949],
                "source_bbox": [320, 1699, 562, 1949],
                "text_pixel_bbox": [320, 1699, 562, 1949],
                "line_polygons": [],
                "confidence": 0.80,
                "tipo": "fala",
                "balloon_type": "textured",
            },
        ]
        page = OutputPage(
            y_top=0,
            y_bottom=2400,
            image=np.full((2400, 760, 3), 255, dtype=np.uint8),
            ocr_result={
                "_vision_blocks": [
                    {
                        "bbox": text["source_bbox"],
                        "text_pixel_bbox": text["text_pixel_bbox"],
                        "line_polygons": text["line_polygons"],
                        "confidence": text["confidence"],
                        "balloon_type": text["balloon_type"],
                    }
                    for text in texts
                ]
            },
            text_layers={"texts": texts},
        )

        changed = _finalize_output_page_ocr_metadata(page, page_number=1)

        self.assertTrue(changed)
        self.assertEqual([text["text"] for text in page.text_layers["texts"]], ["PLEASE, FOR THE CHILD'S SAKE."])
        self.assertEqual(page.text_layers["texts"][0]["translated"], "POR FAVOR, PELO BEM DA CRIANCA.")
        self.assertEqual(len(page.ocr_result["_vision_blocks"]), 1)

    def test_strip_inpainter_prewarm_is_enabled_by_default(self):
        from strip.run import _strip_inpainter_prewarm_enabled

        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(_strip_inpainter_prewarm_enabled())

    def test_strip_inpainter_prewarm_can_be_disabled_by_flag(self):
        from strip.run import _strip_inpainter_prewarm_enabled

        with patch.dict("os.environ", {"TRADUZAI_STRIP_INPAINTER_PREWARM": "0"}, clear=True):
            self.assertFalse(_strip_inpainter_prewarm_enabled())

    def test_koharu_precompute_maps_full_page_ocr_into_band_coordinates(self):
        from strip.run import _split_koharu_page_result_into_bands
        from strip.types import BBox, Balloon, Band, VerticalStrip

        strip = VerticalStrip(
            image=np.zeros((200, 220, 3), dtype=np.uint8),
            width=220,
            height=200,
            source_page_breaks=[0, 100, 200],
            page_x_offsets=[0, 10],
        )
        band = Band(
            y_top=115,
            y_bottom=165,
            balloons=[Balloon(BBox(20, 120, 80, 150), 0.91)],
            strip_slice=np.zeros((50, 220, 3), dtype=np.uint8),
        )
        page_result = {
            "texts": [
                {
                    "id": "koharu_1",
                    "text": "안녕",
                    "bbox": [12, 25, 52, 45],
                    "text_pixel_bbox": [14, 27, 50, 43],
                    "balloon_bbox": [8, 20, 60, 50],
                    "line_polygons": [[[12, 25], [52, 25], [52, 45], [12, 45]]],
                    "ocr_source": "vision-koharu-paddle-ocr-vl-1.5",
                }
            ],
            "_vision_blocks": [
                {
                    "bbox": [8, 20, 60, 50],
                    "text_pixel_bbox": [14, 27, 50, 43],
                    "line_polygons": [[[12, 25], [52, 25], [52, 45], [12, 45]]],
                    "confidence": 0.9,
                }
            ],
            "_vision_backend": "koharu-http",
        }

        mapped = _split_koharu_page_result_into_bands(
            strip,
            page_number=2,
            page_result=page_result,
            page_bands=[(4, band)],
        )

        page = mapped[4]
        text = page["texts"][0]
        self.assertEqual(text["bbox"], [22, 10, 62, 30])
        self.assertEqual(text["text_pixel_bbox"], [24, 12, 60, 28])
        self.assertEqual(text["balloon_bbox"], [18, 5, 70, 35])
        self.assertEqual(text["line_polygons"][0][0], [22, 10])
        self.assertEqual(page["_vision_blocks"][0]["bbox"], [18, 5, 70, 35])
        self.assertEqual(page["_vision_blocks"][0]["text_pixel_bbox"], [24, 12, 60, 28])
        self.assertEqual(page["_vision_blocks"][0]["line_polygons"][0][0], [22, 10])
        self.assertTrue(page["_ocr_stats"]["koharu_cjk_precompute"])

    def test_koharu_precompute_batches_roi_jobs_and_filters_non_translatable_text(self):
        from strip.run import _build_precomputed_koharu_cjk_pages
        from strip.types import BBox, Balloon, Band, VerticalStrip

        strip = VerticalStrip(
            image=np.full((220, 160, 3), 255, dtype=np.uint8),
            width=160,
            height=220,
            source_page_breaks=[0, 110, 220],
            page_x_offsets=[0, 0],
        )
        sfx_band = Band(
            y_top=20,
            y_bottom=80,
            balloons=[Balloon(BBox(20, 25, 110, 70), 0.91)],
            strip_slice=np.full((60, 160, 3), 255, dtype=np.uint8),
        )
        speech_band = Band(
            y_top=140,
            y_bottom=205,
            balloons=[Balloon(BBox(25, 150, 135, 195), 0.94)],
            strip_slice=np.full((65, 160, 3), 255, dtype=np.uint8),
        )
        runtime = MagicMock()

        def fake_batch(jobs, **_kwargs):
            self.assertEqual(len(jobs), 2)
            self.assertTrue(all(job.get("mode") == "roi" for job in jobs))
            return [
                {
                    "texts": [
                        {
                            "text": "......",
                            "bbox": [15, 20, 45, 30],
                            "text_pixel_bbox": [15, 20, 45, 30],
                            "confidence": 0.95,
                            "tipo": "fala",
                        },
                        {
                            "text": "하하..",
                            "bbox": [60, 35, 96, 55],
                            "text_pixel_bbox": [60, 35, 96, 55],
                            "confidence": 0.85,
                            "tipo": "sfx",
                        },
                    ],
                    "_vision_blocks": [{"bbox": [15, 20, 96, 55], "confidence": 0.9}],
                    "_vision_backend": "koharu-http",
                },
                {
                    "texts": [
                        {
                            "text": "도저히 생문을 찾을 수가 없다.",
                            "bbox": [18, 16, 100, 48],
                            "text_pixel_bbox": [18, 16, 100, 48],
                            "confidence": 0.92,
                            "tipo": "fala",
                        }
                    ],
                    "_vision_blocks": [{"bbox": [18, 16, 100, 48], "confidence": 0.92}],
                    "_vision_backend": "koharu-http",
                },
            ]

        runtime.run_koharu_cjk_pages.side_effect = fake_batch
        telemetry = {}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "strip.run._koharu_cjk_strip_precompute_enabled",
            return_value=True,
        ), patch.dict(
            "os.environ",
            {
                "TRADUZAI_KOHARU_CJK_STRIP_ROI": "1",
                "TRADUZAI_KOHARU_CJK_SELECTIVE": "1",
                "TRADUZAI_KOHARU_CJK_EMPTY_ROI_FILTER": "0",
                "TRADUZAI_KOHARU_CJK_PAGE_FALLBACK": "0",
            },
            clear=False,
        ):
            page_paths = [Path(tmp) / "001.jpg", Path(tmp) / "002.jpg"]
            mapped = _build_precomputed_koharu_cjk_pages(
                strip,
                [sfx_band, speech_band],
                runtime,
                page_paths,
                models_dir="N:/TraduzAI/models",
                idioma_origem="ko",
                telemetry=telemetry,
            )

        runtime.run_koharu_cjk_pages.assert_called_once()
        self.assertEqual(mapped[0]["texts"], [])
        self.assertEqual(mapped[0]["_vision_blocks"], [])
        self.assertEqual(mapped[0]["_ocr_stats"]["koharu_cjk_filtered_text_count"], 2)
        self.assertEqual(mapped[1]["texts"][0]["text"], "도저히 생문을 찾을 수가 없다.")
        self.assertEqual(telemetry["batch_mode"], "roi")
        self.assertEqual(telemetry["filtered_text_count"], 2)
        self.assertTrue(all(job.get("known_text_bboxes") for job in runtime.run_koharu_cjk_pages.call_args.args[0]))

    def test_koharu_precompute_can_skip_empty_roi_before_worker(self):
        from strip.run import _build_precomputed_koharu_cjk_pages
        from strip.types import BBox, Balloon, Band, VerticalStrip

        strip = VerticalStrip(
            image=np.full((420, 360, 3), 255, dtype=np.uint8),
            width=360,
            height=420,
            source_page_breaks=[0, 420],
            page_x_offsets=[0],
        )
        band = Band(
            y_top=20,
            y_bottom=390,
            balloons=[Balloon(BBox(40, 70, 320, 340), 0.91)],
            strip_slice=np.full((370, 360, 3), 255, dtype=np.uint8),
        )
        runtime = MagicMock()
        telemetry = {}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "strip.run._koharu_cjk_strip_precompute_enabled",
            return_value=True,
        ), patch(
            "strip.run._koharu_roi_has_textlike_content",
            return_value=(False, "blank_roi"),
        ), patch.dict(
            "os.environ",
            {
                "TRADUZAI_KOHARU_CJK_STRIP_ROI": "1",
                "TRADUZAI_KOHARU_CJK_EMPTY_ROI_FILTER": "1",
            },
            clear=False,
        ):
            mapped = _build_precomputed_koharu_cjk_pages(
                strip,
                [band],
                runtime,
                [Path(tmp) / "001.jpg"],
                models_dir="N:/TraduzAI/models",
                idioma_origem="ko",
                telemetry=telemetry,
            )

        runtime.run_koharu_cjk_pages.assert_not_called()
        self.assertEqual(mapped[0]["texts"], [])
        self.assertEqual(mapped[0]["_ocr_stats"]["koharu_cjk_mode"], "roi_quick_skip")
        self.assertEqual(telemetry["roi_quick_skip_count"], 1)
        self.assertEqual(telemetry["roi_quick_skip_reasons"], {"blank_roi": 1})

    def test_koharu_precompute_keeps_korean_sfx_inside_white_balloon(self):
        from strip.run import _koharu_cjk_text_is_translatable

        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud558\ud558\ud558",
                    "tipo": "fala",
                    "balloon_type": "white",
                    "background_rgb": [255, 255, 255],
                }
            )
        )
        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud06c\uc544\uc544\uc545!!",
                    "tipo": "sfx",
                    "balloon_type": "textured",
                    "background_rgb": [36, 36, 36],
                    "confidence": 0.934,
                },
                idioma_origem="ko",
            )
        )
        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud558\ud558\ud558.",
                    "tipo": "sfx",
                    "balloon_type": "textured",
                    "background_rgb": [36, 36, 36],
                    "confidence": 0.928,
                },
                idioma_origem="ko",
            )
        )
        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud558\ud558\ud558",
                    "tipo": "sfx",
                    "balloon_type": "white",
                    "block_profile": "white_balloon",
                    "background_rgb": [255, 255, 255],
                    "confidence": 0.486,
                },
                idioma_origem="ko",
            )
        )
        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud558\ud558\ud558",
                    "tipo": "sfx",
                    "balloon_type": "textured",
                    "background_rgb": [255, 255, 255],
                    "confidence": 0.486,
                },
                idioma_origem="ko",
            )
        )
        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ud558\ud558\ud558",
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "background_rgb": [36, 36, 36],
                }
            )
        )
        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "\ube44\ud2c0",
                    "tipo": "sfx",
                    "balloon_type": "textured",
                    "background_rgb": [36, 36, 36],
                    "confidence": 0.95,
                },
                idioma_origem="ko",
            )
        )

    def test_koharu_roi_page_fallback_recovers_text_filtered_from_roi(self):
        from strip.run import _build_precomputed_koharu_cjk_pages
        from strip.types import BBox, Balloon, Band, VerticalStrip

        strip = VerticalStrip(
            image=np.full((220, 160, 3), 255, dtype=np.uint8),
            width=160,
            height=220,
            source_page_breaks=[0, 220],
            page_x_offsets=[0],
        )
        band = Band(
            y_top=20,
            y_bottom=180,
            balloons=[Balloon(BBox(20, 25, 140, 150), 0.91)],
            strip_slice=np.full((160, 160, 3), 255, dtype=np.uint8),
        )
        runtime = MagicMock()

        def fake_batch(jobs, **_kwargs):
            if jobs[0]["mode"] == "roi":
                return [
                    {
                        "texts": [
                            {
                                "text": "\ube44\ud2c0",
                                "bbox": [45, 80, 85, 110],
                                "text_pixel_bbox": [45, 80, 85, 110],
                                "confidence": 0.95,
                                "tipo": "sfx",
                                "balloon_type": "textured",
                            }
                        ],
                        "_vision_blocks": [{"bbox": [45, 80, 85, 110], "confidence": 0.95}],
                        "_vision_backend": "koharu-http",
                    }
                ]
            self.assertEqual(jobs[0]["mode"], "page_fallback")
            return [
                {
                    "texts": [
                        {
                            "text": "\uc820\uc7a5!",
                            "bbox": [42, 92, 92, 122],
                            "text_pixel_bbox": [42, 92, 92, 122],
                            "confidence": 0.72,
                            "tipo": "fala",
                            "balloon_type": "white",
                            "block_profile": "white_balloon",
                        }
                    ],
                    "_vision_blocks": [{"bbox": [42, 92, 92, 122], "confidence": 0.72}],
                    "_vision_backend": "koharu-http",
                }
            ]

        runtime.run_koharu_cjk_pages.side_effect = fake_batch
        telemetry = {}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "strip.run._koharu_cjk_strip_precompute_enabled",
            return_value=True,
        ), patch.dict(
            "os.environ",
            {
                "TRADUZAI_KOHARU_CJK_STRIP_ROI": "1",
                "TRADUZAI_KOHARU_CJK_SELECTIVE": "1",
                "TRADUZAI_KOHARU_CJK_EMPTY_ROI_FILTER": "0",
                "TRADUZAI_KOHARU_CJK_PAGE_FALLBACK": "1",
            },
            clear=False,
        ):
            mapped = _build_precomputed_koharu_cjk_pages(
                strip,
                [band],
                runtime,
                [Path(tmp) / "001.jpg"],
                models_dir="N:/TraduzAI/models",
                idioma_origem="ko",
                telemetry=telemetry,
            )

        self.assertEqual(runtime.run_koharu_cjk_pages.call_count, 2)
        self.assertEqual(mapped[0]["texts"][0]["text"], "\uc820\uc7a5!")
        self.assertEqual(telemetry["page_fallback_candidate_count"], 1)
        self.assertEqual(telemetry["page_fallback_text_count"], 1)

    def test_koharu_roi_page_fallback_prioritizes_large_textlike_empty_balloon(self):
        from strip.run import _build_precomputed_koharu_cjk_pages
        from strip.types import BBox, Balloon, Band, VerticalStrip

        strip = VerticalStrip(
            image=np.full((440, 200, 3), 255, dtype=np.uint8),
            width=200,
            height=440,
            source_page_breaks=[0, 220, 440],
            page_x_offsets=[0, 0],
        )
        small_band = Band(
            y_top=20,
            y_bottom=180,
            balloons=[Balloon(BBox(20, 40, 70, 80), 0.91)],
            strip_slice=np.full((160, 200, 3), 255, dtype=np.uint8),
        )
        large_band = Band(
            y_top=240,
            y_bottom=400,
            balloons=[Balloon(BBox(10, 250, 190, 370), 0.91)],
            strip_slice=np.full((160, 200, 3), 255, dtype=np.uint8),
        )
        runtime = MagicMock()

        def fake_batch(jobs, **_kwargs):
            if jobs[0]["mode"] == "roi":
                return [
                    {"texts": [], "_vision_blocks": [], "_vision_backend": "koharu-http"}
                    for _job in jobs
                ]
            self.assertEqual([job["page_number"] for job in jobs], [2])
            return [
                {
                    "texts": [
                        {
                            "text": "\uadf8\uac74...!!",
                            "bbox": [35, 40, 145, 82],
                            "text_pixel_bbox": [35, 40, 145, 82],
                            "confidence": 0.72,
                            "tipo": "fala",
                            "balloon_type": "white",
                            "block_profile": "white_balloon",
                        }
                    ],
                    "_vision_blocks": [{"bbox": [35, 40, 145, 82], "confidence": 0.72}],
                    "_vision_backend": "koharu-http",
                }
            ]

        runtime.run_koharu_cjk_pages.side_effect = fake_batch
        telemetry = {}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "strip.run._koharu_cjk_strip_precompute_enabled",
            return_value=True,
        ), patch(
            "strip.run._koharu_roi_has_textlike_content",
            return_value=(True, "test"),
        ), patch.dict(
            "os.environ",
            {
                "TRADUZAI_KOHARU_CJK_STRIP_ROI": "1",
                "TRADUZAI_KOHARU_CJK_SELECTIVE": "1",
                "TRADUZAI_KOHARU_CJK_EMPTY_ROI_FILTER": "1",
                "TRADUZAI_KOHARU_CJK_PAGE_FALLBACK": "1",
                "TRADUZAI_KOHARU_CJK_PAGE_FALLBACK_MAX": "1",
            },
            clear=False,
        ):
            mapped = _build_precomputed_koharu_cjk_pages(
                strip,
                [small_band, large_band],
                runtime,
                [Path(tmp) / "001.jpg", Path(tmp) / "002.jpg"],
                models_dir="N:/TraduzAI/models",
                idioma_origem="ko",
                telemetry=telemetry,
            )

        self.assertEqual(runtime.run_koharu_cjk_pages.call_count, 2)
        self.assertEqual(mapped[1]["texts"][0]["text"], "\uadf8\uac74...!!")
        self.assertEqual(telemetry["page_fallback_job_count"], 1)
        self.assertEqual(telemetry["page_fallback_text_count"], 1)

    def test_koharu_precompute_skips_short_non_hangul_noise_for_korean_source(self):
        from strip.run import _koharu_cjk_text_is_translatable

        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "おおお",
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "background_rgb": [45, 45, 45],
                },
                idioma_origem="ko",
            )
        )
        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "bloto",
                    "tipo": "fala",
                    "balloon_type": "textured",
                    "background_rgb": [42, 42, 42],
                },
                idioma_origem="ko",
            )
        )
        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "bloto",
                    "tipo": "fala",
                    "balloon_type": "white",
                    "background_rgb": [255, 255, 255],
                },
                idioma_origem="ko",
            )
        )
        self.assertFalse(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "逃生天魔",
                    "tipo": "narracao",
                    "balloon_type": "cover",
                    "background_rgb": [245, 242, 230],
                },
                idioma_origem="ko",
            )
        )
        self.assertTrue(
            _koharu_cjk_text_is_translatable(
                {
                    "text": "정말 훌륭한 무인을 구했군.",
                    "tipo": "fala",
                    "balloon_type": "white",
                    "background_rgb": [255, 255, 255],
                },
                idioma_origem="ko",
            )
        )

    def test_summarize_band_perf_aggregates_smart_skip_shadow_counts(self):
        from strip.run import _summarize_band_perf
        from strip.types import Band

        band = Band(y_top=0, y_bottom=100)
        band.perf = {
            "band_index": 0,
            "text_count": 2,
            "smart_skip_shadow_candidate_count": 1,
            "smart_skip_shadow_not_safe_count": 1,
            "smart_skip_shadow_category_counts": {
                "credit_or_watermark": 1,
                "not_safe_to_skip": 1,
            },
            "durations_sec": {"ocr": 1.0},
            "total_sec": 1.0,
        }

        summary = _summarize_band_perf([band])

        self.assertEqual(summary["smart_skip_shadow_candidate_count"], 1)
        self.assertEqual(summary["smart_skip_shadow_not_safe_count"], 1)
        self.assertEqual(
            summary["smart_skip_shadow_category_counts"],
            {"credit_or_watermark": 1, "not_safe_to_skip": 1},
        )
        self.assertEqual(summary["entries"][0]["smart_skip_shadow_candidate_count"], 1)

    def test_summarize_band_perf_aggregates_smart_skip_real_counts(self):
        from strip.run import _summarize_band_perf
        from strip.types import Band

        band = Band(y_top=0, y_bottom=100)
        band.perf = {
            "band_index": 0,
            "text_count": 2,
            "smart_skip_real_candidate_count": 2,
            "smart_skip_real_not_safe_count": 0,
            "smart_skip_real_applied": True,
            "smart_skip_real_category_counts": {
                "credit_or_watermark": 1,
                "timer_or_ui": 1,
            },
            "durations_sec": {"ocr": 1.0},
            "total_sec": 1.0,
        }

        summary = _summarize_band_perf([band])

        self.assertEqual(summary["smart_skip_real_candidate_count"], 2)
        self.assertEqual(summary["smart_skip_real_not_safe_count"], 0)
        self.assertEqual(summary["smart_skip_real_applied_band_count"], 1)
        self.assertEqual(
            summary["smart_skip_real_category_counts"],
            {"credit_or_watermark": 1, "timer_or_ui": 1},
        )
        self.assertTrue(summary["entries"][0]["smart_skip_real_applied"])

    def test_summarize_band_perf_aggregates_macro_ocr_real_counts(self):
        from strip.run import _summarize_band_perf
        from strip.types import Band

        band = Band(y_top=0, y_bottom=100)
        band.perf = {
            "band_index": 0,
            "text_count": 1,
            "ocr_precomputed_page": True,
            "ocr_runtime_skipped": True,
            "ocr_macro_ocr_real": True,
            "ocr_macro_window_count": 2,
            "ocr_macro_ocr_block_count": 3,
            "ocr_macro_ocr_empty_record_count": 1,
            "durations_sec": {"ocr": 0.01},
            "total_sec": 0.02,
        }

        summary = _summarize_band_perf([band])

        self.assertEqual(summary["ocr_precomputed_page_band_count"], 1)
        self.assertEqual(summary["ocr_runtime_skipped_band_count"], 1)
        self.assertEqual(summary["ocr_macro_ocr_real_band_count"], 1)
        self.assertEqual(summary["ocr_macro_window_count"], 2)
        self.assertEqual(summary["ocr_macro_ocr_block_count"], 3)
        self.assertEqual(summary["ocr_macro_ocr_empty_record_count"], 1)
        self.assertTrue(summary["entries"][0]["ocr_precomputed_page"])

    def test_inpaint_stage_preserves_band_and_source_page_metadata_for_debug(self):
        from strip.process_bands import _run_inpaint_stage
        from strip.types import Band

        captured = {}

        class CapturingInpainter:
            def inpaint_band_image(self, band_rgb, ocr_page):
                captured.update(ocr_page)
                return band_rgb.copy()

        band = Band(
            y_top=120,
            y_bottom=220,
            strip_slice=np.full((100, 160, 3), 255, dtype=np.uint8),
        )
        translated_page = {
            "numero": 99,
            "texts": [{"text": "HELLO", "bbox": [20, 30, 70, 55]}],
            "_vision_blocks": [{"bbox": [20, 30, 70, 55]}],
        }

        _run_inpaint_stage(
            band,
            inpainter=CapturingInpainter(),
            translated_page=translated_page,
            band_index=7,
            source_page_number=3,
        )

        self.assertEqual(captured["_band_index"], 7)
        self.assertEqual(captured["_source_page_number"], 3)
        self.assertEqual(captured["_band_y_top"], 120)

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

    def test_run_chapter_preserves_distinct_original_clean_and_rendered_pages(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"

            def fake_process_band(band, **_kw):
                band.cleaned_slice = np.full_like(band.strip_slice, 200)
                band.rendered_slice = np.full_like(band.strip_slice, 32)
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            self.assertEqual(len(output_pages), 1)
            page = output_pages[0]
            self.assertTrue(hasattr(page, "original_image"))
            self.assertTrue(hasattr(page, "inpainted_image"))
            self.assertIsNotNone(page.original_image)
            self.assertIsNotNone(page.inpainted_image)
            self.assertFalse(np.array_equal(page.original_image, page.inpainted_image))
            self.assertFalse(np.array_equal(page.inpainted_image, page.image))

    def test_run_chapter_attaches_strip_perf_summary_to_first_page_profile(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"
            chapter_telemetry = {}

            def fake_process_band(band, **_kw):
                idx = len(getattr(fake_process_band, "seen", []))
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                band.perf = {
                    "band_index": idx,
                    "text_count": 1,
                    "fast_white_balloon_count": 1,
                    "fast_local_balloon_count": 1,
                    "remaining_inpaint_blocks": 0,
                    "fast_white_rejection_reasons": {"no_white_fill_mask": 1 + idx},
                    "fast_local_rejection_reasons": {"no_flat_fill": 2},
                    "ocr_full_page_mapped": 1,
                    "ocr_crop_fallback_attempts": 2,
                    "ocr_crop_fallback_recovered": 1,
                    "ocr_quick_skipped_no_text": idx == 1,
                    "ocr_scanlation_credit_skipped": idx == 0,
                    "ocr_cover_editorial_skipped": idx == 1,
                    "unchanged_translation_skip": idx == 0,
                    "skip_processing_copy": idx == 1,
                    "durations_sec": {
                        "ocr": 0.1 + (0.2 * idx),
                        "translate": 0.2,
                        "inpaint": 0.5 - (0.2 * idx),
                        "typeset": 0.4,
                    },
                    "total_sec": 1.0 + idx,
                }
                fake_process_band.seen = [*getattr(fake_process_band, "seen", []), True]
                return band

            with patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                    chapter_telemetry=chapter_telemetry,
                )

            summary = output_pages[0].page_profile.get("strip_perf_summary")
            self.assertEqual(summary["band_count"], 2)
            self.assertEqual(summary["text_count"], 2)
            self.assertEqual(summary["fast_white_balloon_count"], 2)
            self.assertEqual(summary["fast_local_balloon_count"], 2)
            self.assertEqual(summary["fast_white_band_count"], 2)
            self.assertEqual(summary["remaining_inpaint_blocks"], 0)
            self.assertEqual(summary["fast_white_rejection_reasons"], {"no_white_fill_mask": 3})
            self.assertEqual(summary["fast_local_rejection_reasons"], {"no_flat_fill": 4})
            self.assertEqual(summary["ocr_full_page_mapped"], 2)
            self.assertEqual(summary["ocr_crop_fallback_attempts"], 4)
            self.assertEqual(summary["ocr_crop_fallback_recovered"], 2)
            self.assertEqual(summary["ocr_quick_skipped_no_text_band_count"], 1)
            self.assertEqual(summary["ocr_scanlation_credit_skipped_band_count"], 1)
            self.assertEqual(summary["ocr_cover_editorial_skipped_band_count"], 1)
            self.assertEqual(summary["unchanged_translation_skip_band_count"], 1)
            self.assertEqual(summary["skip_processing_copy_band_count"], 1)
            self.assertAlmostEqual(summary["durations_sec"]["ocr"], 0.4)
            self.assertEqual(len(summary["entries"]), 2)
            self.assertEqual([entry["band_index"] for entry in summary["entries"]], [0, 1])
            self.assertEqual(summary["top_bands"][0]["band_index"], 1)
            self.assertEqual(summary["top_ocr_bands"][0]["band_index"], 1)
            self.assertTrue(summary["top_inpaint_bands"][0]["unchanged_translation_skip"])
            self.assertEqual(summary["top_inpaint_bands"][0]["band_index"], 0)
            self.assertEqual(summary["top_typeset_bands"][0]["durations_sec"]["typeset"], 0.4)
            self.assertIn("top_bands", summary)
            self.assertIn("chapter_stage_durations_sec", summary)
            self.assertIn("strip_build", summary["chapter_stage_durations_sec"])
            self.assertIn("strip_process_bands_total", summary["chapter_stage_durations_sec"])
            self.assertGreater(chapter_telemetry["wall_total_sec"], 0)
            self.assertEqual(output_pages[0].ocr_result["page_profile"], output_pages[0].page_profile)

    def test_run_chapter_attaches_macro_ocr_shadow_when_enabled(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            shadow_report = {
                "status": "PASS",
                "window_mode": "band-groups",
                "macro_window_count": 1,
                "text_line_count": 1,
            }

            def fake_process_band(band, **_kw):
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {
                    "texts": [{"text": "HELLO", "bbox": [10, 10, 60, 40]}],
                    "_vision_blocks": [{"bbox": [10, 10, 60, 40]}],
                }
                return band

            with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR_SHADOW": "1"}, clear=False), patch(
                "strip.run.process_band",
                side_effect=fake_process_band,
            ), patch("strip.run._run_macro_ocr_shadow", return_value=shadow_report) as shadow:
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            shadow.assert_called_once()
            self.assertEqual(output_pages[0].page_profile["macro_ocr_shadow"], shadow_report)
            self.assertEqual(
                output_pages[0].ocr_result["page_profile"]["macro_ocr_shadow"],
                shadow_report,
            )

    def test_macro_ocr_shadow_uses_vision_stack_engine_when_runtime_has_no_engine(self):
        from strip.run import _run_macro_ocr_shadow
        from strip.types import OutputPage

        class FakeOcr:
            def recognize_blocks_from_page(self, _image, blocks, **_kwargs):
                self._last_recognize_blocks_stats = {
                    "block_count": len(blocks),
                    "full_page_mapped": len(blocks),
                    "crop_fallback_attempts": 0,
                    "crop_fallback_recovered": 0,
                }
                return [{"text": "HELLO"} for _ in blocks]

        page = OutputPage(
            y_top=0,
            y_bottom=100,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            text_layers={"texts": [{"text": "HELLO", "bbox": [10, 10, 60, 40]}]},
            inpaint_blocks=[{"bbox": [10, 10, 60, 40]}],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()) as get_ocr:
            report = _run_macro_ocr_shadow([page], runtime=object(), idioma_origem="en")

        get_ocr.assert_called_once_with("max", lang="en")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["text_line_count"], 1)

    def test_macro_ocr_shadow_reports_fallback_resolved_metrics(self):
        from strip.run import _run_macro_ocr_shadow
        from strip.types import OutputPage

        class FakeOcr:
            def __init__(self):
                self.outputs = [
                    [{"text": "HELLO"}, {"text": "TEMPERATURE 23 DEGREES CELSIUS,"}],
                    [{"text": "00:00:05"}],
                ]

            def recognize_blocks_from_page(self, _image, blocks, **_kwargs):
                self._last_recognize_blocks_stats = {
                    "block_count": len(blocks),
                    "full_page_mapped": len(blocks),
                    "crop_fallback_attempts": 0,
                    "crop_fallback_recovered": 0,
                }
                return self.outputs.pop(0)

        page = OutputPage(
            y_top=0,
            y_bottom=220,
            image=np.zeros((220, 120, 3), dtype=np.uint8),
            original_image=np.zeros((220, 120, 3), dtype=np.uint8),
            text_layers={
                "texts": [
                    {"text": "HELLO", "bbox": [10, 10, 50, 35]},
                    {"text": "TEMPERATURE DEGREES CELSIUS,", "bbox": [10, 55, 80, 85]},
                    {"text": "oo:oo:os", "bbox": [10, 170, 70, 195]},
                ]
            },
            inpaint_blocks=[
                {"bbox": [10, 10, 50, 35]},
                {"bbox": [10, 55, 80, 85]},
                {"bbox": [10, 170, 70, 195]},
            ],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()):
            report = _run_macro_ocr_shadow([page], runtime=object(), idioma_origem="en")

        self.assertEqual(report["numeric_token_change_count"], 1)
        self.assertEqual(report["numeric_confusable_variation_count"], 1)
        self.assertEqual(report["fallback_required_count"], 1)
        self.assertEqual(report["fallback_resolved_different_count"], 1)
        self.assertEqual(report["fallback_resolved_different_text_rate"], 0.3333)
        self.assertEqual(report["fallback_adjusted_ocr_call_count"], 3)

    def test_macro_ocr_shadow_can_gate_on_fallback_resolved_text(self):
        from strip.run import _run_macro_ocr_shadow
        from strip.types import OutputPage

        class FakeOcr:
            def recognize_blocks_from_page(self, _image, blocks, **_kwargs):
                self._last_recognize_blocks_stats = {
                    "block_count": len(blocks),
                    "full_page_mapped": len(blocks),
                    "crop_fallback_attempts": 0,
                    "crop_fallback_recovered": 0,
                }
                return [{"text": "HELLO"}, {"text": "TEMPERATURE 23 DEGREES CELSIUS,"}]

        page = OutputPage(
            y_top=0,
            y_bottom=100,
            image=np.zeros((100, 120, 3), dtype=np.uint8),
            original_image=np.zeros((100, 120, 3), dtype=np.uint8),
            text_layers={
                "texts": [
                    {"text": "HELLO", "bbox": [10, 10, 50, 35]},
                    {"text": "TEMPERATURE DEGREES CELSIUS,", "bbox": [10, 55, 80, 85]},
                ]
            },
            inpaint_blocks=[
                {"bbox": [10, 10, 50, 35]},
                {"bbox": [10, 55, 80, 85]},
            ],
        )

        with patch("vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()):
            default_report = _run_macro_ocr_shadow([page], runtime=object(), idioma_origem="en")
        with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR_GATE_FALLBACK_RESOLVED": "1"}), patch(
            "vision_stack.runtime._get_ocr_engine", return_value=FakeOcr()
        ):
            fallback_resolved_report = _run_macro_ocr_shadow(
                [page], runtime=object(), idioma_origem="en"
            )

        self.assertEqual(default_report["status"], "FAIL")
        self.assertEqual(default_report["different_text_rate"], 0.5)
        self.assertFalse(default_report["gate_on_fallback_resolved_text"])
        self.assertEqual(fallback_resolved_report["status"], "PASS")
        self.assertEqual(fallback_resolved_report["text_quality_gate_rate"], 0.0)
        self.assertTrue(fallback_resolved_report["gate_on_fallback_resolved_text"])

    def test_run_chapter_forwards_translation_runtime_options_to_bands(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            records = []
            translation_context = {"memory": [{"source": "HELLO", "target": "OLA"}]}

            def fake_process_band(band, **kw):
                records.append(kw)
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                    models_dir="D:/traduzai_data/models",
                    ollama_host="http://127.0.0.1:11435",
                    ollama_model="custom-translator",
                    translation_context=translation_context,
                )

            self.assertEqual(records[0]["models_dir"], "D:/traduzai_data/models")
            self.assertEqual(records[0]["ollama_host"], "http://127.0.0.1:11435")
            self.assertEqual(records[0]["ollama_model"], "custom-translator")
            self.assertIs(records[0]["translation_context"], translation_context)

    def test_run_chapter_passes_source_page_number_for_each_band(self):
        from strip.run import _source_page_number_for_band, run_chapter
        from strip.types import Band, VerticalStrip

        strip = VerticalStrip(
            image=np.zeros((900, 200, 3), dtype=np.uint8),
            width=200,
            height=900,
            source_page_breaks=[0, 300, 600, 900],
        )
        self.assertEqual(_source_page_number_for_band(strip, Band(y_top=40, y_bottom=120)), 1)
        self.assertEqual(_source_page_number_for_band(strip, Band(y_top=330, y_bottom=410)), 2)
        self.assertEqual(_source_page_number_for_band(strip, Band(y_top=560, y_bottom=650)), 3)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 3)
            output = tmp_path / "out"
            records = []

            def fake_process_band(band, **kw):
                records.append(kw["source_page_number"])
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=3,
                    detector=_make_detector_with_n_balloons(3),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            self.assertEqual(records, [1, 2, 3])

    def test_run_chapter_passes_precomputed_macro_ocr_page_when_enabled(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            records = []
            runtime = MagicMock()
            runtime._get_ocr_engine.return_value = object()

            def fake_process_band(band, **kw):
                records.append(kw)
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR": "1"}, clear=False), patch(
                "ocr.macro_ocr.recognize_macro_ocr_windows",
                return_value=(["HELLO"], {"macro_window_count": 1, "full_page_mapped": 1}, []),
            ) as macro_ocr, patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=runtime,
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            self.assertEqual(macro_ocr.call_count, 1)
            self.assertIsNotNone(records[0].get("precomputed_ocr_page"))
            precomputed = records[0]["precomputed_ocr_page"]
            self.assertEqual(precomputed["texts"][0]["text"], "HELLO")
            self.assertTrue(precomputed["_ocr_stats"]["macro_ocr_real"])
            self.assertEqual(precomputed["_ocr_stats"]["macro_window_count"], 1)

    def test_macro_ocr_precompute_respects_scanlation_credit_skip(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            records = []
            runtime = MagicMock()
            runtime._get_ocr_engine.return_value = object()

            def fake_process_band(band, **kw):
                records.append(kw)
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR": "1"}, clear=False), patch(
                "vision_stack.runtime._looks_like_scanlation_credit_band",
                return_value=True,
            ), patch(
                "ocr.macro_ocr.recognize_macro_ocr_windows",
                return_value=(["SHOULD NOT RUN"], {"macro_window_count": 1}, []),
            ) as macro_ocr, patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=runtime,
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            macro_ocr.assert_not_called()
            self.assertIsNone(records[0].get("precomputed_ocr_page"))

    def test_macro_ocr_precompute_respects_cover_editorial_skip(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            records = []
            runtime = MagicMock()
            runtime._get_ocr_engine.return_value = object()

            def fake_process_band(band, **kw):
                records.append(kw)
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR": "1"}, clear=False), patch(
                "vision_stack.runtime._looks_like_scanlation_credit_band",
                return_value=False,
            ), patch(
                "vision_stack.runtime._looks_like_cover_editorial_band",
                return_value=True,
            ), patch(
                "ocr.macro_ocr.recognize_macro_ocr_windows",
                return_value=(["SHOULD NOT RUN"], {"macro_window_count": 1}, []),
            ) as macro_ocr, patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=runtime,
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            macro_ocr.assert_not_called()
            self.assertIsNone(records[0].get("precomputed_ocr_page"))

    def test_run_chapter_reports_macro_ocr_precompute_time_when_enabled(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            runtime = MagicMock()
            runtime._get_ocr_engine.return_value = object()

            def fake_process_band(band, **_kw):
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict("os.environ", {"TRADUZAI_MACRO_OCR": "1"}, clear=False), patch(
                "ocr.macro_ocr.recognize_macro_ocr_windows",
                return_value=(["HELLO"], {"macro_window_count": 1, "full_page_mapped": 1}, []),
            ), patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=runtime,
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            summary = output_pages[0].page_profile["strip_perf_summary"]
            self.assertTrue(summary["macro_ocr_precompute"]["enabled"])
            self.assertEqual(summary["macro_ocr_precompute"]["precomputed_band_count"], 1)
            self.assertIn("macro_ocr_precompute", summary["durations_sec"])

    def test_macro_ocr_precompute_can_skip_pages_below_min_block_threshold(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            records = []
            runtime = MagicMock()
            runtime._get_ocr_engine.return_value = object()

            def fake_process_band(band, **kw):
                records.append(kw)
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict(
                "os.environ",
                {
                    "TRADUZAI_MACRO_OCR": "1",
                    "TRADUZAI_MACRO_OCR_PRECOMPUTE_MIN_BLOCKS": "2",
                },
                clear=False,
            ), patch(
                "ocr.macro_ocr.recognize_macro_ocr_windows",
                return_value=(["SHOULD NOT RUN"], {"macro_window_count": 1}, []),
            ) as macro_ocr, patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=runtime,
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            macro_ocr.assert_not_called()
            self.assertIsNone(records[0].get("precomputed_ocr_page"))
            summary = output_pages[0].page_profile["strip_perf_summary"]
            self.assertEqual(summary["macro_ocr_precompute"]["skipped_page_count"], 1)
            self.assertEqual(
                summary["macro_ocr_precompute"]["skip_reasons"],
                {"below_min_blocks": 1},
            )

    def test_run_chapter_starts_inpainter_prewarm_before_processing_bands(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            events = []
            inpainter = MagicMock()
            inpainter.prewarm_band_inpainter.side_effect = lambda: events.append("prewarm")

            def fake_process_band(band, **_kw):
                events.append("process")
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict("os.environ", {"TRADUZAI_STRIP_INPAINTER_PREWARM": "1"}, clear=False), patch(
                "strip.run.process_band",
                side_effect=fake_process_band,
            ):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=_make_detector_with_n_balloons(1),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=inpainter,
                    typesetter=MagicMock(),
                )

            self.assertIn("prewarm", events)
            self.assertLess(events.index("prewarm"), events.index("process"))

    def test_run_chapter_starts_inpainter_prewarm_before_strip_detection(self):
        from strip.run import run_chapter
        from strip.detect_balloons import detect_strip_balloons as real_detect_strip_balloons

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 1)
            output = tmp_path / "out"
            detector = _make_detector_with_n_balloons(1)
            events = []

            def fake_start_prewarm(*_args, **_kw):
                events.append("prewarm")
                return None

            def fake_detect(_strip, detector):
                events.append("detect")
                return real_detect_strip_balloons(_strip, detector=detector)

            def fake_process_band(band, **_kw):
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch("strip.run._start_inpainter_prewarm", side_effect=fake_start_prewarm), patch(
                "strip.run.detect_strip_balloons",
                side_effect=fake_detect,
            ), patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=1,
                    detector=detector,
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            self.assertEqual(events[:2], ["prewarm", "detect"])

    def test_run_chapter_reports_scheduler_executor_when_enabled(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"
            process_order = []

            def fake_process_band(band, **kw):
                process_order.append(kw["page_idx"])
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict(
                "os.environ",
                {"TRADUZAI_STRIP_SCHEDULER_EXECUTOR": "1"},
                clear=False,
            ), patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            summary = output_pages[0].page_profile["strip_perf_summary"]
            scheduler_executor = summary["scheduler_executor"]
            self.assertTrue(scheduler_executor["enabled"])
            self.assertEqual(scheduler_executor["mode"], "sequential_safe")
            self.assertEqual(scheduler_executor["processed_band_count"], 2)
            self.assertEqual(scheduler_executor["task_count"], 10)
            self.assertEqual(scheduler_executor["max_gpu_parallel"], 1)
            self.assertEqual(process_order, [0, 1])

    def test_run_chapter_overlap_executor_starts_next_band_after_translate_release(self):
        import threading
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"
            records = []
            band1_started = threading.Event()

            def fake_process_band(band, **kw):
                idx = kw["page_idx"]
                records.append({
                    "idx": idx,
                    "glossario": dict(kw.get("glossario") or {}),
                    "has_gpu_lock": kw.get("gpu_stage_lock") is not None,
                })
                if idx == 0:
                    callback = kw.get("ordered_context_after_translate_callback")
                    self.assertIsNotNone(callback)
                    callback({
                        "texts": [{"id": "early", "translated": "CEDO"}],
                        "_glossary_additions": {"EARLY": "cedo"},
                    })
                    self.assertTrue(
                        band1_started.wait(1.0),
                        "overlap executor did not start band 1 before band 0 finished",
                    )
                elif idx == 1:
                    band1_started.set()
                band.cleaned_slice = band.strip_slice.copy()
                band.rendered_slice = band.strip_slice.copy()
                band.ocr_result = {"texts": [], "_vision_blocks": []}
                return band

            with patch.dict(
                "os.environ",
                {"TRADUZAI_STRIP_SCHEDULER_EXECUTOR": "overlap"},
                clear=False,
            ), patch("strip.run.process_band", side_effect=fake_process_band):
                output_pages = run_chapter(
                    image_files=files,
                    output_dir=output,
                    target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                )

            summary = output_pages[0].page_profile["strip_perf_summary"]
            scheduler_executor = summary["scheduler_executor"]
            self.assertEqual(scheduler_executor["mode"], "overlap_context_release")
            self.assertEqual(scheduler_executor["processed_band_count"], 2)
            self.assertTrue(all(record["has_gpu_lock"] for record in records))
            self.assertEqual(records[1]["glossario"]["EARLY"], "cedo")


class RunningHistoryTests(unittest.TestCase):
    """H.4 — history rolante de bandas passado para contextual reviewer."""

    def test_ordered_band_context_snapshot_copies_and_limits_history(self):
        from strip.run import _build_ordered_band_context_snapshot

        running_history = [{"idx": idx} for idx in range(25)]
        running_glossary = {"BASE": "base"}

        snapshot = _build_ordered_band_context_snapshot(running_history, running_glossary)
        kwargs = snapshot.to_process_kwargs()

        self.assertEqual(len(kwargs["band_history"]), 20)
        self.assertEqual(kwargs["band_history"][0]["idx"], 5)
        self.assertEqual(kwargs["glossario"], {"BASE": "base"})

        kwargs["band_history"][0]["idx"] = "mutated"
        kwargs["glossario"]["BASE"] = "mutated"
        self.assertEqual(running_history[5]["idx"], 5)
        self.assertEqual(running_glossary["BASE"], "base")

    def test_ordered_band_context_merge_copies_result_and_glossary_additions(self):
        from strip.run import _merge_ordered_band_context_after_commit

        running_history = []
        running_glossary = {"BASE": "base"}
        ocr_result = {
            "texts": [{"id": "t1", "translated": "OLA"}],
            "_glossary_additions": {"FENRIS": "Fenris"},
        }

        _merge_ordered_band_context_after_commit(
            running_history,
            running_glossary,
            ocr_result,
        )
        ocr_result["texts"][0]["translated"] = "MUTATED"
        ocr_result["_glossary_additions"]["FENRIS"] = "MUTATED"

        self.assertEqual(running_history[0]["texts"][0]["translated"], "OLA")
        self.assertEqual(running_glossary["FENRIS"], "Fenris")

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

    def test_ordered_context_can_merge_after_translate_callback_without_double_merge(self):
        from strip.run import run_chapter

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            files = _write_pages(tmp_path, 2)
            output = tmp_path / "out"
            records = []

            def fake_process_band(band, **kw):
                idx = len(records)
                records.append({
                    "band_history": list(kw.get("band_history") or []),
                    "glossario": dict(kw.get("glossario") or {}),
                })
                callback = kw.get("ordered_context_after_translate_callback")
                if idx == 0 and callback:
                    callback({
                        "texts": [{"id": "early", "translated": "CEDO"}],
                        "_glossary_additions": {"EARLY": "cedo"},
                    })
                    band.ocr_result = {
                        "texts": [{"id": "late", "translated": "TARDE"}],
                        "_glossary_additions": {"LATE": "tarde"},
                    }
                else:
                    band.ocr_result = {"texts": [], "_vision_blocks": []}
                band.rendered_slice = band.strip_slice.copy()
                return band

            with patch("strip.run.process_band", side_effect=fake_process_band):
                run_chapter(
                    image_files=files, output_dir=output, target_count=2,
                    detector=_make_detector_with_n_balloons(2),
                    runtime=MagicMock(), translator=MagicMock(),
                    inpainter=MagicMock(), typesetter=MagicMock(),
                    glossario={"BASE": "base"},
                )

            self.assertEqual(records[1]["glossario"]["EARLY"], "cedo")
            self.assertNotIn("LATE", records[1]["glossario"])
            self.assertEqual(records[1]["band_history"][0]["texts"][0]["id"], "early")
