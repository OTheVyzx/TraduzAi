import io
import sys
import unittest
import tempfile
import json
import importlib
import contextlib
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from typesetter.renderer import build_render_blocks  # noqa: E402


class MainEmitTests(unittest.TestCase):
    def setUp(self) -> None:
        main._EMIT_STDOUT_FAILED = False

    def tearDown(self) -> None:
        main._detach_work_dir_log_handler()

    def test_emit_swallow_oserror_from_stdout_once(self) -> None:
        stderr = io.StringIO()

        with patch("builtins.print", side_effect=OSError(22, "Invalid argument")):
            with patch.object(main.sys, "stderr", stderr):
                main.emit("progress", message="primeira")
                main.emit("progress", message="segunda")

        log_output = stderr.getvalue()
        self.assertIn("Falha ao emitir evento JSON no stdout", log_output)
        self.assertEqual(log_output.count("Falha ao emitir evento JSON no stdout"), 1)

    def test_main_lists_supported_languages_in_cli_mode(self) -> None:
        stdout = io.StringIO()

        with patch.object(main.sys, "argv", ["main.py", "--list-supported-languages"]):
            with patch("translator.translate.list_supported_google_languages", return_value=[{"code": "en", "label": "English"}]):
                with patch.object(main.sys, "stdout", stdout):
                    main.main()

        self.assertEqual(stdout.getvalue().strip(), '[{"code": "en", "label": "English"}]')

    def test_main_prints_help_instead_of_trying_to_open_help_as_config(self) -> None:
        stdout = io.StringIO()

        with patch.object(main.sys, "argv", ["main.py", "--help"]):
            with patch.object(main.sys, "stdout", stdout):
                main.main()

        output = stdout.getvalue()
        self.assertIn("--list-supported-languages", output)
        self.assertIn("--translate-page", output)
        self.assertIn("--input", output)

    def test_runner_cli_parser_accepts_mock_debug_flags(self) -> None:
        parsed = main._parse_runner_cli_args(
            [
                "--input",
                "fixtures/tiny_chapter/original",
                "--work",
                "The Regressed Mercenary Has a Plan",
                "--target",
                "pt-BR",
                "--mode",
                "mock",
                "--debug",
                "--skip-inpaint",
                "--skip-ocr",
                "--strict",
                "--export-mode",
                "clean",
                "--output",
                "debug/runs/tiny_chapter",
            ]
        )

        self.assertEqual(parsed["source_path"], "fixtures/tiny_chapter/original")
        self.assertEqual(parsed["obra"], "The Regressed Mercenary Has a Plan")
        self.assertEqual(parsed["idioma_destino"], "pt-BR")
        self.assertEqual(parsed["mode"], "mock")
        self.assertTrue(parsed["debug"])
        self.assertTrue(parsed["skip_inpaint"])
        self.assertTrue(parsed["skip_ocr"])
        self.assertTrue(parsed["strict"])
        self.assertEqual(parsed["export_mode"], "clean")

    def test_runner_cli_mock_generates_project_and_reports_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "original"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            (input_dir / "001.png").write_bytes(b"fake image")

            exit_code = main._run_pipeline_runner_cli(
                {
                    "source_path": str(input_dir),
                    "obra": "Fixture",
                    "idioma_origem": "en",
                    "idioma_destino": "pt-BR",
                    "mode": "mock",
                    "debug": True,
                    "strict": False,
                    "export_mode": "clean",
                    "work_dir": str(output_dir),
                }
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "project.json").exists())
            self.assertTrue((output_dir / "qa_report.json").exists())
            self.assertTrue((output_dir / "qa_report.md").exists())
            self.assertTrue((output_dir / "issues.csv").exists())

    def test_runner_cli_strict_returns_nonzero_when_mock_has_critical_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "original"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            (input_dir / "001.png").write_bytes(b"fake image")

            exit_code = main._run_pipeline_runner_cli(
                {
                    "source_path": str(input_dir),
                    "obra": "Fixture",
                    "idioma_origem": "en",
                    "idioma_destino": "pt-BR",
                    "mode": "mock",
                    "debug": True,
                    "strict": True,
                    "mock_critical": True,
                    "work_dir": str(output_dir),
                }
            )

            self.assertNotEqual(exit_code, 0)
            qa = json.loads((output_dir / "qa_report.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["summary"]["critical"], 1)

    def test_select_local_venv_python_prefers_project_venv_when_current_is_global(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_root = Path(tmp)
            venv_python = script_root / "venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")

            selected = main._select_local_venv_python(
                current_executable=str(script_root / "global-python.exe"),
                script_root=script_root,
            )

        self.assertEqual(selected, venv_python.resolve())

    def test_select_local_venv_python_returns_none_when_already_using_project_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script_root = Path(tmp)
            venv_python = script_root / "venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")

            selected = main._select_local_venv_python(
                current_executable=str(venv_python),
                script_root=script_root,
            )

        self.assertIsNone(selected)

    def test_configure_pipeline_logging_uses_info_level_and_force(self) -> None:
        with patch.object(main.logging, "basicConfig") as basic_config:
            main._configure_pipeline_logging()

        basic_config.assert_called_once()
        kwargs = basic_config.call_args.kwargs
        self.assertEqual(kwargs["level"], main.logging.INFO)
        self.assertTrue(kwargs["force"])

    def test_attach_work_dir_log_handler_writes_pipeline_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            main._configure_pipeline_logging()
            log_path = main._attach_work_dir_log_handler(tmp)
            logger = main.logging.getLogger("tests.pipeline")
            logger.info("decisao de teste")
            for handler in main.logging.getLogger().handlers:
                with contextlib.suppress(Exception):
                    handler.flush()

            self.assertTrue(log_path.exists())
            self.assertIn("decisao de teste", log_path.read_text(encoding="utf-8"))

    def test_decision_trace_records_entries_and_writes_qa_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")

            decision_log.configure_decision_trace(tmp)
            decision_log.record_decision(
                stage="ocr",
                action="drop_block",
                reason="punctuation_only",
                page=27,
                layer="ocr_001",
                text="-",
            )
            decision_log.record_decision(
                stage="layout",
                action="clamp_balloon_bbox",
                reason="top_caption_overexpand",
                page=41,
                layer="ocr_001",
                details={"expansion_ratio": 9.9},
            )
            decision_log.finalize_decision_trace({"obra": "Teste"})

            trace_path = Path(tmp) / "decision_trace.jsonl"
            qa_path = Path(tmp) / "qa_report.json"

            self.assertTrue(trace_path.exists())
            self.assertTrue(qa_path.exists())

            lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 2)

            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            self.assertEqual(qa["summary"]["total_decisions"], 2)
            self.assertEqual(qa["summary"]["by_action"]["drop_block"], 1)
            self.assertEqual(qa["summary"]["by_reason"]["top_caption_overexpand"], 1)
            self.assertEqual(qa["flagged_pages"], [27, 41])

    def test_run_translate_page_maps_original_field_and_persists_returned_translations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            translated_dir = Path(tmp) / "translated"
            translated_dir.mkdir()
            (translated_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "idioma_origem": "en",
                "idioma_destino": "pt-BR",
                "obra": "Teste",
                "contexto": {"sinopse": "abc"},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [
                            {
                                "id": "layer-1",
                                "bbox": [1, 2, 3, 4],
                                "original": "HELLO",
                                "traduzido": "",
                                "tipo": "fala",
                            }
                        ],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            def fake_translate_pages(**kwargs):
                texts = kwargs["ocr_results"][0]["texts"]
                self.assertEqual(texts[0]["text"], "HELLO")
                return [
                    {
                        "texts": [
                            {
                                **texts[0],
                                "translated": "OLÁ",
                                "traduzido": "OLÁ",
                            }
                        ]
                    }
                ]

            with patch("translator.translate.translate_pages", side_effect=fake_translate_pages):
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_translate_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["paginas"][0]["text_layers"][0]["translated"], "OLÁ")
            self.assertEqual(saved["paginas"][0]["text_layers"][0]["traduzido"], "OLÁ")
            self.assertEqual(saved["paginas"][0]["textos"][0]["translated"], "OLÁ")

    def test_run_detect_page_uses_vision_worker_and_persists_editor_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "_models_dir": "D:/traduzai_data/models",
                "_vision_worker_path": "D:/TraduzAi/vision-worker/target/debug/traduzai-vision.exe",
                "_ollama_host": "http://localhost:11434",
                "idioma_origem": "en",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "image_layers": {
                            "base": {"key": "base", "path": "originals/001.jpg", "visible": True, "locked": True},
                            "rendered": {"key": "rendered", "path": "translated/001.jpg", "visible": True, "locked": True},
                        },
                        "text_layers": [],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            ocr_page = {
                "texts": [
                    {
                        "id": "tl_001_001",
                        "bbox": [10, 20, 110, 120],
                        "balloon_bbox": [8, 18, 118, 130],
                        "text_pixel_bbox": [12, 24, 106, 116],
                        "text": "HELLO",
                        "confidence": 0.93,
                        "ocr_source": "vision-paddleocr",
                        "tipo": "fala",
                        "detector": "comic-text-bubble-detector",
                        "layout_group_size": 2,
                        "balloon_subregions": [[8, 18, 60, 130], [62, 18, 118, 130]],
                        "layout_profile": "connected_balloon",
                    }
                ],
                "_vision_blocks": [{"bbox": [10, 20, 110, 120], "confidence": 0.88}],
            }

            def fake_run_ocr(image_path, models_dir, vision_worker_path, idioma_origem):
                self.assertEqual(Path(image_path), originals_dir / "001.jpg")
                self.assertEqual(models_dir, "D:/traduzai_data/models")
                self.assertEqual(vision_worker_path, project["_vision_worker_path"])
                self.assertEqual(idioma_origem, "en")
                return ocr_page

            with patch("ocr.detector.run_ocr", side_effect=fake_run_ocr):
                with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_: page):
                    with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                        with patch("main.render_page_image"):
                            with patch("main.emit_progress"):
                                with patch("main.emit"):
                                    main._run_detect_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["_work_dir"], str(project_path.parent))
            self.assertEqual(len(saved["paginas"][0]["text_layers"]), 1)
            layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(layer["original"], "HELLO")
            self.assertEqual(layer["translated"], "")
            self.assertEqual(layer["source_bbox"], [10, 20, 110, 120])
            self.assertEqual(layer["balloon_bbox"], [8, 18, 118, 130])
            self.assertEqual(layer["text_pixel_bbox"], [12, 24, 106, 116])
            self.assertEqual(layer["layout_group_size"], 2)
            self.assertEqual(layer["layout_profile"], "connected_balloon")
            self.assertEqual(saved["paginas"][0]["textos"][0]["layout_group_size"], 2)
            self.assertEqual(saved["paginas"][0]["textos"][0]["balloon_bbox"], [8, 18, 118, 130])
            self.assertEqual(saved["paginas"][0]["textos"][0]["text_pixel_bbox"], [12, 24, 106, 116])
            self.assertEqual(saved["paginas"][0]["textos"][0]["ocr_source"], "vision-paddleocr")
            self.assertEqual(saved["paginas"][0]["inpaint_blocks"][0]["bbox"], [10, 20, 110, 120])
            self.assertEqual(saved["paginas"][0]["inpaint_blocks"][0]["confidence"], 0.88)

    def test_sync_page_legacy_aliases_preserves_rich_text_metadata(self) -> None:
        page = {
            "image_layers": {
                "base": {"path": "originals/001.jpg"},
                "rendered": {"path": "translated/001.jpg"},
            },
            "text_layers": [
                {
                    "id": "tl_001_001",
                    "bbox": [20, 30, 180, 120],
                    "source_bbox": [24, 34, 176, 116],
                    "text_pixel_bbox": [40, 52, 150, 96],
                    "line_polygons": [[[40, 52], [150, 52], [150, 96], [40, 96]]],
                    "original": "HELLO",
                    "translated": "OLA",
                    "tipo": "fala",
                    "ocr_confidence": 0.93,
                    "ocr_source": "vision-paddleocr",
                    "balloon_bbox": [12, 18, 190, 130],
                    "balloon_type": "white",
                }
            ],
        }

        main._sync_page_legacy_aliases(page)

        legacy = page["textos"][0]
        self.assertEqual(legacy["id"], "tl_001_001")
        self.assertEqual(legacy["text_pixel_bbox"], [40, 52, 150, 96])
        self.assertEqual(legacy["source_bbox"], [24, 34, 176, 116])
        self.assertEqual(legacy["line_polygons"][0][2], [150, 96])
        self.assertEqual(legacy["ocr_source"], "vision-paddleocr")
        self.assertEqual(legacy["ocr_confidence"], 0.93)
        self.assertEqual(legacy["confianca_ocr"], 0.93)
        self.assertEqual(legacy["balloon_type"], "white")

    def test_run_ocr_page_uses_detected_blocks_when_text_layers_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")
            (translated_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "inpaint_blocks": [
                            {"bbox": [10, 20, 110, 120], "confidence": 0.88},
                            {"bbox": [130, 140, 230, 280], "confidence": 0.72},
                        ],
                        "text_layers": [],
                        "textos": [],
                    }
                ]
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            calls = []

            def fake_run_ocr_on_block(image_path, bbox):
                calls.append((Path(image_path), list(bbox)))
                if bbox == [10, 20, 110, 120]:
                    return ("HELLO", 0.91)
                return ("BYE", 0.83)

            with patch("ocr.detector.run_ocr_on_block", side_effect=fake_run_ocr_on_block):
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_ocr_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            page = saved["paginas"][0]
            self.assertEqual(len(page["text_layers"]), 2)
            self.assertEqual(page["text_layers"][0]["source_bbox"], [10, 20, 110, 120])
            self.assertEqual(page["text_layers"][0]["layout_bbox"], [10, 20, 110, 120])
            self.assertEqual(page["text_layers"][0]["original"], "HELLO")
            self.assertEqual(page["text_layers"][0]["translated"], "")
            self.assertEqual(page["text_layers"][0]["ocr_confidence"], 0.91)
            self.assertEqual(page["text_layers"][1]["original"], "BYE")
            self.assertEqual(page["textos"][0]["original"], "HELLO")
            self.assertEqual(calls, [
                (originals_dir / "001.jpg", [10, 20, 110, 120]),
                (originals_dir / "001.jpg", [130, 140, 230, 280]),
            ])

    def test_normalize_text_layer_repairs_local_text_pixel_bbox_and_polygons(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "tl_009_001",
                "tipo": "fala",
                "source_bbox": [422, 1365, 769, 1536],
                "layout_bbox": [422, 1365, 769, 1536],
                "balloon_bbox": [422, 1365, 769, 1536],
                "text_pixel_bbox": [436, 30, 760, 180],
                "line_polygons": [[[436, 30], [760, 30], [760, 180], [436, 180]]],
                "original": "HOW COULD YOU?!",
                "translated": "COMO VOCE PODE?!",
            },
            page_number=9,
            layer_index=0,
        )

        self.assertEqual(layer["text_pixel_bbox"], [436, 1395, 760, 1545])
        self.assertEqual(layer["line_polygons"][0][0], [436, 1395])
        self.assertEqual(layer["line_polygons"][0][2], [760, 1545])

    def test_carry_translations_for_detected_layers_matches_by_geometry_not_index(self) -> None:
        existing_layers = [
            {
                "original": "DAMMIT. WE LET OUR GUARD DOWN!",
                "translated": "DROGA...",
                "text_pixel_bbox": [448, 936, 865, 1100],
            },
            {
                "original": "XEV",
                "translated": "XEV",
                "text_pixel_bbox": [849, 1484, 985, 1626],
            },
            {
                "original": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!",
                "translated": "HA MUITO MAIS...",
                "text_pixel_bbox": [411, 2085, 735, 2192],
            },
            {
                "original": "AT THIS RATE, WE'RE ALL GONNA DIE-",
                "translated": "NESSE RITMO...",
                "text_pixel_bbox": [676, 2226, 900, 2334],
            },
            {
                "original": "COMMANDER!",
                "translated": "COMANDANTE!",
                "text_pixel_bbox": [363, 2981, 667, 3055],
            },
        ]
        reviewed_texts = [
            {"text": "KEUK?!", "text_pixel_bbox": [580, 286, 721, 350]},
            {"text": "DAMMIT. WE LET OUR GUARD DOWN!", "text_pixel_bbox": [428, 908, 881, 1114]},
            {"text": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!", "text_pixel_bbox": [411, 2085, 735, 2192]},
            {"text": "AT THIS RATE, WE'RE ALL GONNA DIE-", "text_pixel_bbox": [676, 2226, 900, 2334]},
            {"text": "COMMANDER!", "text_pixel_bbox": [362, 2977, 669, 3056]},
        ]

        carried = main._carry_translations_for_detected_layers(existing_layers, reviewed_texts)

        self.assertEqual(carried[0], "")
        self.assertEqual(carried[1], "DROGA...")
        self.assertEqual(carried[2], "HA MUITO MAIS...")
        self.assertEqual(carried[3], "NESSE RITMO...")
        self.assertEqual(carried[4], "COMANDANTE!")

    def test_carry_translations_for_detected_layers_combines_translations_for_merged_ocr_text(self) -> None:
        existing_layers = [
            {
                "original": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!",
                "translated": "HA MUITO MAIS...",
                "text_pixel_bbox": [411, 2085, 735, 2192],
            },
            {
                "original": "AT THIS RATE, WE'RE ALL GONNA DIE-",
                "translated": "NESSE RITMO...",
                "text_pixel_bbox": [676, 2226, 900, 2334],
            },
        ]
        reviewed_texts = [
            {
                "text": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO! AT THIS RATE, WE'RE ALL GONNA DIE-",
                "text_pixel_bbox": [411, 2085, 901, 2334],
            }
        ]

        carried = main._carry_translations_for_detected_layers(existing_layers, reviewed_texts)

        self.assertEqual(carried, ["HA MUITO MAIS... NESSE RITMO..."])

    def test_carry_translations_for_detected_layers_splits_old_merged_translation_into_new_layers(self) -> None:
        existing_layers = [
            {
                "original": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO! AT THIS RATE, WE'RE ALL GONNA DIE-",
                "translated": "HÁ MUITO MAIS DELES DO QUE FOI RELATADO TAMBÉM! NESSE RITMO, TODOS NÓS VAMOS MORRER-",
                "text_pixel_bbox": [411, 2085, 901, 2334],
            }
        ]
        reviewed_texts = [
            {
                "text": "AT THIS RATE, WE'RE ALL GONNA DIE—",
                "text_pixel_bbox": [675, 2226, 901, 2334],
            },
            {
                "text": "THERE ARE WAY MORE OF THEM THAN WAS REPORTED TOO!",
                "text_pixel_bbox": [411, 2085, 735, 2193],
            },
        ]

        carried = main._carry_translations_for_detected_layers(existing_layers, reviewed_texts)

        self.assertEqual(
            carried,
            [
                "NESSE RITMO, TODOS NÓS VAMOS MORRER-",
                "HÁ MUITO MAIS DELES DO QUE FOI RELATADO TAMBÉM!",
            ],
        )

    def test_run_process_block_uses_layout_bbox_when_legacy_bbox_alias_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [
                            {
                                "id": "tl_001_001",
                                "source_bbox": [100, 200, 300, 400],
                                "layout_bbox": [110, 220, 310, 430],
                                "original": "",
                                "translated": "",
                                "tipo": "fala",
                                "style": {
                                    "fonte": "ComicNeue-Bold.ttf",
                                    "tamanho": 20,
                                    "cor": "#FFFFFF",
                                    "cor_gradiente": [],
                                    "contorno": "#000000",
                                    "contorno_px": 2,
                                    "glow": False,
                                    "glow_cor": "",
                                    "glow_px": 0,
                                    "sombra": False,
                                    "sombra_cor": "",
                                    "sombra_offset": [0, 0],
                                    "bold": False,
                                    "italico": False,
                                    "rotacao": 0,
                                    "alinhamento": "center",
                                },
                            }
                        ],
                    }
                ]
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            bbox_calls = []

            def fake_run_ocr_on_block(image_path, bbox):
                bbox_calls.append(list(bbox))
                return ("TEXT", 0.95)

            with patch("ocr.detector.run_ocr_on_block", side_effect=fake_run_ocr_on_block):
                with patch("inpainter.lama.run_inpainting"):
                    with patch("main.render_page_image"):
                        with patch("main.emit_progress"):
                            with patch("main.emit"):
                                main._run_process_block(project_path, 0, "tl_001_001", "ocr")

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(bbox_calls, [[110, 220, 310, 430]])
            self.assertEqual(layer["original"], "TEXT")
            self.assertEqual(layer["bbox"], [110, 220, 310, 430])
            self.assertEqual(layer["ocr_confidence"], 0.95)

    def test_build_text_layer_preserves_connected_balloon_metadata_for_renderer(self) -> None:
        ocr_text = {
            "id": "tl_001_003",
            "text": "IT MAY BE NOTHING MORE THAN A HALF-FINISHED CULTIVATION METHOD...",
            "bbox": [113, 1513, 705, 1767],
            "balloon_bbox": [113, 1513, 705, 1767],
            "balloon_subregions": [[113, 1513, 395, 1767], [409, 1513, 705, 1767]],
            "tipo": "fala",
            "confidence": 0.97,
            "layout_group_size": 1,
            "connected_balloon_orientation": "left-right",
            "connected_detection_confidence": 1.0,
            "connected_group_confidence": 0.898,
            "connected_position_confidence": 0.952,
            "subregion_confidence": 1.0,
            "connected_text_groups": [[113, 1513, 373, 1722], [432, 1558, 705, 1767]],
            "connected_lobe_bboxes": [[113, 1513, 395, 1767], [409, 1513, 705, 1767]],
            "connected_position_bboxes": [[113, 1513, 345, 1712], [462, 1568, 705, 1767]],
            "connected_focus_bboxes": [[113, 1513, 345, 1712], [462, 1568, 705, 1767]],
            "estilo": {
                "fonte": "ComicNeue-Bold.ttf",
                "tamanho": 48,
                "cor": "#111111",
                "contorno": "#FFFFFF",
                "contorno_px": 2,
                "alinhamento": "center",
            },
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=2,
            ocr_text=ocr_text,
            translated=(
                "PODE SER NADA MAIS DO QUE UM METODO DE CULTIVO INACABADO, "
                "MAS SEUS EFEITOS SAO MAIS DO QUE SUFICIENTES. UM PODER QUE "
                "PERMITE SUPERAR SEUS PROPRIOS LIMITES EM UM INSTANTE."
            ),
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        blocks = build_render_blocks([layer])

        self.assertEqual(layer.get("connected_position_bboxes"), ocr_text["connected_position_bboxes"])
        self.assertEqual(layer.get("connected_text_groups"), ocr_text["connected_text_groups"])
        self.assertEqual(layer.get("connected_detection_confidence"), 1.0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].get("balloon_subregions"), ocr_text["balloon_subregions"])
        self.assertEqual(blocks[0].get("connected_position_bboxes"), ocr_text["connected_position_bboxes"])

    def test_build_text_layer_preserves_entity_metadata_for_project_json(self) -> None:
        layer = main.build_text_layer(
            page_number=2,
            layer_index=1,
            ocr_text={
                "id": "tl_002_002",
                "text": "GHHISLAN PERDIUM",
                "bbox": [10, 20, 120, 90],
                "balloon_bbox": [8, 18, 124, 98],
                "tipo": "fala",
                "confidence": 0.88,
                "entity_flags": ["source_entity_repaired", "glossary_locked"],
                "entity_repairs": [
                    {
                        "phase": "source",
                        "kind": "character",
                        "from": "GHHISLAN PERDIUM",
                        "to": "Ghislain Perdium",
                    }
                ],
                "glossary_hits": [
                    {
                        "phase": "target",
                        "source": "Mana Core",
                        "target": "Núcleo de Mana",
                    }
                ],
                "qa_flags": ["entity_suspect"],
            },
            translated="GHISLAIN PERDIUM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["entity_flags"], ["source_entity_repaired", "glossary_locked"])
        self.assertEqual(layer["entity_repairs"][0]["to"], "Ghislain Perdium")
        self.assertEqual(layer["glossary_hits"][0]["target"], "Núcleo de Mana")
        self.assertEqual(layer["qa_flags"], ["entity_suspect"])


    def test_sync_page_legacy_aliases_prefers_render_bbox(self) -> None:
        page = {
            "image_layers": {
                "base": {"path": "originals/001.jpg"},
                "rendered": {"path": "translated/001.jpg"},
            },
            "text_layers": [
                {
                    "id": "tl_001_001",
                    "render_bbox": [14, 18, 72, 86],
                    "layout_bbox": [10, 12, 80, 90],
                    "source_bbox": [10, 12, 80, 90],
                    "tipo": "fala",
                    "original": "HELLO",
                    "translated": "OLÁ",
                    "ocr_confidence": 0.91,
                    "style": {
                        "fonte": "ComicNeue-Bold.ttf",
                        "tamanho": 20,
                        "cor": "#FFFFFF",
                        "cor_gradiente": [],
                        "contorno": "#000000",
                        "contorno_px": 2,
                        "glow": False,
                        "glow_cor": "",
                        "glow_px": 0,
                        "sombra": False,
                        "sombra_cor": "",
                        "sombra_offset": [0, 0],
                        "bold": False,
                        "italico": False,
                        "rotacao": 0,
                        "alinhamento": "center",
                    },
                }
            ],
        }

        main._sync_page_legacy_aliases(page)

        self.assertEqual(page["textos"][0]["bbox"], [14, 18, 72, 86])


if __name__ == "__main__":
    unittest.main()
