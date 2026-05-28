import io
import sys
import unittest
import tempfile
import json
import importlib
import contextlib
from pathlib import Path
from types import SimpleNamespace
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

    def test_emit_falls_back_to_ascii_json_on_unicode_encode_error(self) -> None:
        calls = []

        def fake_print(value, *args, **kwargs):
            calls.append(value)
            if len(calls) == 1:
                raise UnicodeEncodeError("cp1252", "第", 0, 1, "character maps to <undefined>")

        with patch("builtins.print", side_effect=fake_print):
            main.emit("error", message="Arquivo não encontrado: 第270话.cbz")

        self.assertEqual(len(calls), 2)
        self.assertIn("\\u7b2c270\\u8bdd.cbz", calls[1])

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

    def test_prepare_inpaint_base_final_text_box_cleanup_is_disabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(main._final_render_text_box_cleanup_enabled())

        with patch.dict("os.environ", {}, clear=True):
            result = main._prepare_inpaint_base_for_render(
                original_path=Path("missing-original.jpg"),
                inpainted_path=Path("missing-inpaint.jpg"),
                texts=[
                    {
                        "original": "WHAT",
                        "translated": "O QUE",
                        "bbox": [10, 10, 40, 20],
                        "balloon_type": "white",
                    }
                ],
            )

        self.assertEqual(result, Path("missing-inpaint.jpg"))

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

    def test_glossary_used_report_shadow_collects_hits_and_blocks(self) -> None:
        report = main.build_glossary_used_report(
            {"obra": "Demo", "idioma_origem": "ko", "idioma_destino": "pt-BR", "glossario": {"A": "B"}},
            {"personagens": ["Hero"], "memoria_lexical": {"Clan": "Cla"}},
            [
                {
                    "texts": [
                        {
                            "id": "t1",
                            "original": "src",
                            "glossary_hits": [{"source": "A", "target": "B"}],
                            "translation_blocked_text": "texto original",
                            "qa_flags": ["translation_fallback_phrase"],
                        }
                    ]
                }
            ],
        )

        self.assertEqual(report["mode"], "shadow")
        self.assertEqual(report["summary"]["entry_count"], 3)
        self.assertEqual(report["summary"]["used_hit_count"], 1)
        self.assertEqual(report["summary"]["blocked_translation_count"], 1)
        self.assertEqual(report["qa_flags"]["translation_fallback_phrase"], 1)

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
            project = json.loads((output_dir / "project.json").read_text(encoding="utf-8"))
            self.assertEqual(project["qa"]["timing"]["sidecar_path"], "performance_timing.json")
            self.assertGreater(project["qa"]["timing"]["total_sec"], 0)

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

    def test_runner_cli_mock_critical_persists_blocked_preview_without_path_migration(self) -> None:
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
                    "mock_critical": True,
                    "work_dir": str(output_dir),
                }
            )

            self.assertEqual(exit_code, 0)
            project = json.loads((output_dir / "project.json").read_text(encoding="utf-8"))
            self.assertEqual(project["output_review_state"], "blocked_preview")
            self.assertEqual(project["qa"]["export_gate"]["status"], "BLOCK")
            self.assertEqual(project["paginas"][0]["arquivo_traduzido"], "translated/001.png")
            self.assertTrue((output_dir / "translated" / "001.png").exists())
            self.assertFalse((output_dir / "translated" / "approved").exists())
            self.assertFalse((output_dir / "translated" / "blocked_preview").exists())

    def test_runner_cli_strict_mock_emits_error_as_last_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "original"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            (input_dir / "001.png").write_bytes(b"fake image")
            emitted = []

            with patch.object(main, "emit", side_effect=lambda msg_type, **kwargs: emitted.append({"type": msg_type, **kwargs})):
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

            self.assertEqual(exit_code, 2)
            self.assertEqual(emitted[-1]["type"], "error")
            self.assertNotIn("complete", [event["type"] for event in emitted])

    def test_build_strip_inpainter_honors_skip_inpaint_without_calling_real_inpaint(self) -> None:
        calls = []

        def real_inpaint(band_rgb, ocr_page):
            calls.append((band_rgb, ocr_page))
            return "inpainted"

        inpainter = main._build_strip_inpainter_for_config({"skip_inpaint": True}, real_inpaint)
        page = {"texts": [{"id": "ocr_001"}]}
        result = inpainter.inpaint_band_image("original-band", page)

        self.assertEqual(result, "original-band")
        self.assertEqual(calls, [])
        self.assertTrue(page["_skip_inpaint_honored"])
        self.assertFalse(page["_strip_used_real_inpaint"])

    def test_runner_cli_writes_engine_preset_to_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "original"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            (input_dir / "001.png").write_bytes(b"fake image")

            with patch.object(main, "_run_pipeline") as run_pipeline:
                exit_code = main._run_pipeline_runner_cli(
                    {
                        "source_path": str(input_dir),
                        "obra": "Fixture",
                        "idioma_origem": "ko",
                        "idioma_destino": "pt-BR",
                        "engine_preset_id": "manhwa_manhua_ocr_guided",
                        "mode": "real",
                        "debug": True,
                        "strict": False,
                        "work_dir": str(output_dir),
                    }
                )

            self.assertEqual(exit_code, 0)
            run_pipeline.assert_called_once()
            config = json.loads((output_dir / "runner_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["engine_preset_id"], "manhwa_manhua_ocr_guided")

    def test_runner_cli_writes_strict_to_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "original"
            output_dir = Path(tmp) / "out"
            input_dir.mkdir()
            (input_dir / "001.png").write_bytes(b"fake image")

            with patch.object(main, "_run_pipeline") as run_pipeline:
                exit_code = main._run_pipeline_runner_cli(
                    {
                        "source_path": str(input_dir),
                        "obra": "Fixture",
                        "idioma_origem": "en",
                        "idioma_destino": "pt-BR",
                        "engine_preset_id": "manga",
                        "mode": "real",
                        "debug": True,
                        "strict": True,
                        "work_dir": str(output_dir),
                    }
                )

            self.assertEqual(exit_code, 0)
            run_pipeline.assert_called_once()
            config = json.loads((output_dir / "runner_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["strict"])

    def test_debug_export_gate_artifacts_write_consistency_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-test")
            project = {
                "qa": {
                    "summary": {
                        "critical_count": 1,
                        "highest_severity": "critical",
                    },
                    "export_gate": {
                        "status": "BLOCK",
                        "critical_issue_count": 1,
                        "review_issue_count": 0,
                        "issues": [
                            {
                                "page": 1,
                                "layer": "t1",
                                "type": "p0_render_blocker",
                                "severity": "critical",
                                "flags": ["bbox_overreach_critical"],
                            }
                        ],
                    },
                }
            }

            consistency = main._write_debug_export_gate_artifacts(recorder, project)

            root = Path(tmp) / "debug" / "e2e"
            consistency_path = root / "11_qa_export_gate" / "qa_export_gate_consistency.json"
            report_path = root / "13_report" / "debug_report.md"
            saved = json.loads(consistency_path.read_text(encoding="utf-8"))

            self.assertTrue(consistency["consistent"])
            self.assertTrue(saved["consistency"]["critical_count_matches"])
            self.assertTrue(report_path.exists())

    def test_debug_export_gate_rewrites_render_plan_final_from_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp) / "debug" / "e2e"
            stale_plan = root / "09_typeset" / "render_plan_final.jsonl"
            stale_plan.parent.mkdir(parents=True, exist_ok=True)
            stale_plan.write_text(
                json.dumps(
                    {
                        "stage": "typeset",
                        "source": "stale",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_042",
                        "page_id": "page_003",
                        "band_id": "page_003_band_042",
                        "coordinate_space": "page",
                        "balloon_bbox": [343, 11293, 538, 11317],
                        "safe_text_box": [343, 11293, 538, 11317],
                        "render_bbox": [350, 11300, 530, 11310],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            recorder = DebugRecorder(Path(tmp), enabled=True, run_id="run-test")
            project = {
                "paginas": [
                    {
                        "numero": 3,
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_003_band_042",
                                "band_id": "page_003_band_042",
                                "balloon_bbox": [0, 10726, 800, 11410],
                                "safe_text_box": [343, 11293, 538, 11317],
                                "render_bbox": [350, 11300, 530, 11310],
                                "original": "HELLO",
                                "translated": "OLA",
                                "qa_flags": ["bbox_overreach_critical"],
                            }
                        ],
                    }
                ],
                "qa": {
                    "summary": {"critical_count": 1, "highest_severity": "critical"},
                    "export_gate": {
                        "status": "BLOCK",
                        "critical_issue_count": 1,
                        "review_issue_count": 0,
                        "issues": [],
                    },
                },
            }

            consistency = main._write_debug_export_gate_artifacts(recorder, project)

            rows = [
                json.loads(line)
                for line in stale_plan.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            sync_audit = json.loads(
                (root / "09_typeset" / "render_plan_final_sync.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "project_json_final")
            self.assertEqual(rows[0]["page_id"], "page_003")
            self.assertEqual(rows[0]["band_id"], "page_003_band_042")
            self.assertEqual(rows[0]["balloon_bbox"], [0, 10726, 800, 11410])
            self.assertEqual(rows[0]["safe_text_box"], [343, 11293, 538, 11317])
            self.assertEqual(rows[0]["render_bbox"], [350, 11300, 530, 11310])
            self.assertEqual(rows[0]["qa_flags"], ["bbox_overreach_critical"])
            self.assertEqual(consistency["render_plan_sync"]["written_count"], 1)
            self.assertEqual(sync_audit["summary"]["written_count"], 1)

    def test_normalize_project_render_balloon_bboxes_uses_renderer_target(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 4,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "trace_id": "ocr_001@page_004_band_071",
                            "translated": "PI",
                            "balloon_bbox": [174, 8816, 225, 8869],
                            "safe_text_box": [184, 8821, 276, 8864],
                            "render_bbox": [198, 8826, 261, 8859],
                            "_render_debug": {
                                "target_bbox": [174, 8816, 285, 8869],
                                "position_bbox": [174, 8816, 285, 8869],
                            },
                        }
                    ],
                }
            ]
        }

        fixed = main._normalize_project_render_balloon_bboxes(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(fixed, 1)
        self.assertEqual(layer["balloon_bbox"], [174, 8816, 285, 8869])
        self.assertEqual(layer["_original_balloon_bbox_before_render_sync"], [174, 8816, 225, 8869])
        self.assertTrue(layer["_balloon_bbox_synced_from_render_debug"])

    def test_normalize_project_render_balloon_bboxes_leaves_valid_balloon_unchanged(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 1,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "balloon_bbox": [10, 10, 100, 80],
                            "safe_text_box": [20, 20, 90, 70],
                            "render_bbox": [30, 30, 80, 60],
                            "_render_debug": {"target_bbox": [0, 0, 500, 500]},
                        }
                    ],
                }
            ]
        }

        fixed = main._normalize_project_render_balloon_bboxes(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(fixed, 0)
        self.assertEqual(layer["balloon_bbox"], [10, 10, 100, 80])
        self.assertNotIn("_balloon_bbox_synced_from_render_debug", layer)

    def test_load_json_file_accepts_utf8_bom_from_windows_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text('{"work_dir":"D:/tmp/traduzai"}', encoding="utf-8-sig")

            loaded = main._load_json_file(config_path)

        self.assertEqual(loaded["work_dir"], "D:/tmp/traduzai")

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

    def test_maybe_reexec_local_venv_preserves_child_exit_code(self) -> None:
        completed = type("Completed", (), {"returncode": 2})()

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(main, "_select_local_venv_python", return_value=Path("venv/Scripts/python.exe")):
                with patch.object(main.subprocess, "run", return_value=completed) as run:
                    with self.assertRaises(SystemExit) as raised:
                        main._maybe_reexec_local_venv()

        self.assertEqual(raised.exception.code, 2)
        run.assert_called_once()

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

    def test_decision_trace_qa_report_includes_export_gate_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            decision_log = importlib.import_module("utils.decision_log")

            decision_log.configure_decision_trace(tmp)
            decision_log.record_decision(
                stage="layout",
                action="flag_block",
                reason="render_outside_balloon",
                page=3,
                layer="ocr_001",
            )
            decision_log.finalize_decision_trace(
                {
                    "obra": "Teste",
                    "qa_summary": {
                        "highest_severity": "critical",
                        "critical_issue_count": 1,
                    },
                    "export_gate": {
                        "status": "BLOCK",
                        "allowed": False,
                        "issue_count": 1,
                        "critical_issue_count": 1,
                        "critical_flag_count": 1,
                        "review_issue_count": 0,
                        "review_flag_count": 0,
                        "needs_review": False,
                        "issues": [
                            {
                                "type": "p0_render_blocker",
                                "severity": "critical",
                                "flags": ["render_outside_balloon"],
                                "trace_id": "ocr_001@page_003_band_001",
                            }
                        ],
                    },
                }
            )

            qa = json.loads((Path(tmp) / "qa_report.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["summary"]["export_gate_status"], "BLOCK")
            self.assertEqual(qa["summary"]["critical_issue_count"], 1)
            self.assertEqual(qa["export_gate"]["status"], "BLOCK")
            self.assertEqual(qa["issues"][0]["type"], "p0_render_blocker")
            self.assertTrue(qa["needs_review"])

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
                "qa": {"summary": {"total": 0}},
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
                        "qa_flags": ["low_ocr_confidence"],
                    }
                ],
                "_vision_blocks": [{"bbox": [10, 20, 110, 120], "confidence": 0.88}],
            }

            def fake_run_ocr(
                image_path,
                models_dir,
                vision_worker_path,
                idioma_origem,
                engine_preset_id="",
                **kwargs,
            ):
                self.assertEqual(Path(image_path), originals_dir / "001.jpg")
                self.assertEqual(models_dir, "D:/traduzai_data/models")
                self.assertEqual(vision_worker_path, project["_vision_worker_path"])
                self.assertEqual(idioma_origem, "en")
                self.assertEqual(engine_preset_id, project.get("engine_preset_id", ""))
                self.assertFalse(kwargs.get("work_title_user_provided", False))
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
            self.assertEqual(layer["balloon_subregions"], [[8, 18, 60, 130], [62, 18, 118, 130]])
            self.assertEqual(saved["paginas"][0]["textos"][0]["layout_group_size"], 2)
            self.assertEqual(saved["paginas"][0]["textos"][0]["balloon_bbox"], [8, 18, 118, 130])
            self.assertEqual(saved["paginas"][0]["textos"][0]["text_pixel_bbox"], [12, 24, 106, 116])
            self.assertEqual(saved["paginas"][0]["textos"][0]["balloon_subregions"], [[8, 18, 60, 130], [62, 18, 118, 130]])
            self.assertEqual(saved["paginas"][0]["textos"][0]["ocr_source"], "vision-paddleocr")
            self.assertEqual(saved["paginas"][0]["inpaint_blocks"][0]["bbox"], [10, 20, 110, 120])
            self.assertEqual(saved["paginas"][0]["inpaint_blocks"][0]["confidence"], 0.88)
            self.assertEqual(saved["qa"]["summary"]["total"], 1)
            self.assertEqual(saved["qa"]["summary"]["counts"], {"low_ocr_confidence": 1})

    def test_run_detect_page_with_default_preset_keeps_bubble_layout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "idioma_origem": "en",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
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
                        "bbox": [12, 24, 90, 80],
                        "text": "HELLO",
                        "confidence": 0.91,
                        "tipo": "fala",
                    }
                ],
                "_vision_blocks": [{"bbox": [12, 24, 90, 80], "confidence": 0.86}],
                "_bubble_regions": [{"bbox": [8, 18, 100, 92], "confidence": 0.77}],
            }

            def fake_run_ocr(image_path, models_dir, vision_worker_path, idioma_origem, engine_preset_id="", **_kwargs):
                self.assertEqual(Path(image_path), originals_dir / "001.jpg")
                self.assertEqual(idioma_origem, "ja")
                self.assertEqual(engine_preset_id, "default")
                return ocr_page

            def fake_enrich(page):
                self.assertEqual(page["_bubble_regions"][0]["bbox"], [8, 18, 100, 92])
                enriched = dict(page)
                enriched["texts"] = [dict(page["texts"][0], balloon_bbox=[8, 18, 100, 92], layout_profile="bubble_region")]
                return enriched

            with patch("ocr.detector.run_ocr", side_effect=fake_run_ocr):
                with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_: page):
                    with patch("layout.balloon_layout.enrich_page_layout", side_effect=fake_enrich) as enrich:
                        with patch("main.render_page_image"):
                            with patch("main.emit_progress"):
                                with patch("main.emit"):
                                    main._run_detect_page(
                                        project_path,
                                        0,
                                        None,
                                        {"idioma_origem": "ja", "engine_preset_id": "default"},
                                    )

            enrich.assert_called_once()
            saved = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["idioma_origem"], "ja")
            self.assertEqual(saved["engine_preset_id"], "default")
            layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(layer["balloon_bbox"], [8, 18, 100, 92])
            self.assertEqual(layer["layout_profile"], "bubble_region")

    def test_run_detect_boxes_page_updates_only_inpaint_blocks(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            Image.new("RGB", (32, 24), "white").save(originals_dir / "001.png")

            existing_layer = {
                "id": "tl_001_001",
                "bbox": [2, 2, 16, 12],
                "source_bbox": [2, 2, 16, 12],
                "layout_bbox": [2, 2, 16, 12],
                "original": "KEEP ME",
                "translated": "MANTER",
            }
            project = {
                "idioma_origem": "en",
                "engine_preset_id": "manga",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.png",
                        "arquivo_traduzido": "translated/001.png",
                        "text_layers": [existing_layer],
                        "textos": [existing_layer],
                        "inpaint_blocks": [{"bbox": [20, 2, 28, 10], "confidence": 0.5}],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            class FakeDetector:
                def __init__(self) -> None:
                    self.calls: list[tuple[tuple[int, ...], float]] = []

                def detect(self, image, conf_threshold=0.5):
                    self.calls.append((tuple(image.shape), conf_threshold))
                    return [SimpleNamespace(xyxy=(1, 3, 18, 14), confidence=0.87)]

            detector = FakeDetector()
            emitted: list[dict] = []

            with patch("vision_stack.runtime._get_detector", return_value=detector) as get_detector:
                with patch("vision_stack.runtime._profile_to_detection_threshold", return_value=0.42):
                    with patch("main.render_page_image") as render_page_image:
                        with patch("main.emit_progress"):
                            with patch.object(main, "emit", side_effect=lambda msg_type, **kwargs: emitted.append({"type": msg_type, **kwargs})):
                                main._run_detect_boxes_page(
                                    project_path,
                                    0,
                                    None,
                                    {"idioma_origem": "ko", "engine_preset_id": "manhwa_manhua"},
                                )

            get_detector.assert_called_once_with("max")
            self.assertEqual(detector.calls, [((24, 32, 3), 0.42)])
            render_page_image.assert_not_called()

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            page = saved["paginas"][0]
            self.assertEqual(saved["idioma_origem"], "ko")
            self.assertEqual(saved["engine_preset_id"], "manhwa_manhua")
            self.assertEqual(page["text_layers"][0]["original"], "KEEP ME")
            self.assertEqual(page["textos"][0]["original"], "KEEP ME")
            self.assertEqual(page["inpaint_blocks"], [{"bbox": [1, 3, 18, 14], "confidence": 0.87, "source_bbox": [1, 3, 18, 14], "text_pixel_bbox": [1, 3, 18, 14]}])
            self.assertEqual(emitted[-1]["type"], "complete")
            self.assertTrue(emitted[-1]["output_path"].replace("\\", "/").endswith("translated/001.png"))

    def test_run_detect_page_cache_is_reused_without_live_ocr_on_second_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.png").write_bytes(b"fake")

            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.png",
                        "arquivo_traduzido": "translated/001.png",
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
                        "text": "HELLO",
                        "confidence": 0.93,
                        "tipo": "fala",
                    }
                ],
                "_vision_blocks": [{"bbox": [10, 20, 110, 120], "confidence": 0.88}],
            }
            run_ocr_calls = {"count": 0}

            def fake_run_ocr(
                image_path,
                models_dir,
                vision_worker_path,
                idioma_origem,
                engine_preset_id="",
                **kwargs,
            ):
                run_ocr_calls["count"] += 1
                self.assertEqual(Path(image_path), originals_dir / "001.png")
                self.assertEqual(idioma_origem, "en")
                self.assertEqual(engine_preset_id, "")
                self.assertFalse(kwargs.get("work_title_user_provided", False))
                return ocr_page

            with patch("ocr.detector.run_ocr", side_effect=fake_run_ocr):
                with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_: page):
                    with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                        with patch("main.render_page_image") as render_page:
                            with patch("main.emit_progress") as emit_progress:
                                with patch("main.emit"):
                                    main._run_detect_page(
                                        project_path,
                                        0,
                                        None,
                                        {"idioma_origem": "en", "engine_preset_id": ""},
                                    )
                                    main._run_detect_page(
                                        project_path,
                                        0,
                                        None,
                                        {"idioma_origem": "en", "engine_preset_id": ""},
                                    )

            self.assertEqual(run_ocr_calls["count"], 1)
            self.assertEqual(render_page.call_count, 2)
            progress_messages = [call.kwargs.get("message") for call in emit_progress.call_args_list]
            self.assertIn("Aplicando deteccao em cache...", progress_messages)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            page = saved["paginas"][0]
            self.assertEqual(page["text_layers"][0]["original"], "HELLO")
            self.assertEqual(page["inpaint_blocks"][0]["bbox"], [10, 20, 110, 120])

    def test_run_detect_page_region_bypasses_detect_ocr_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.png").write_bytes(b"fake")

            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.png",
                        "arquivo_traduzido": "translated/001.png",
                        "text_layers": [],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            run_ocr_calls = {"count": 0}

            def fake_run_ocr(*_args, **_kwargs):
                run_ocr_calls["count"] += 1
                return {
                    "texts": [
                        {
                            "id": f"tl_001_{run_ocr_calls['count']:03}",
                            "bbox": [10, 20, 110, 120],
                            "balloon_bbox": [8, 18, 118, 130],
                            "text": f"HELLO {run_ocr_calls['count']}",
                            "confidence": 0.93,
                            "tipo": "fala",
                        }
                    ],
                    "_vision_blocks": [{"bbox": [10, 20, 110, 120], "confidence": 0.88}],
                }

            with patch("ocr.detector.run_ocr", side_effect=fake_run_ocr):
                with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_: page):
                    with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                        with patch("main.render_page_image"):
                            with patch("main.emit_progress"):
                                with patch("main.emit"):
                                    main._run_detect_page(project_path, 0)
                                    main._run_detect_page(project_path, 0, region={"bbox": [0, 0, 200, 200]})

            self.assertEqual(run_ocr_calls["count"], 2)
            saved = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["paginas"][0]["text_layers"][0]["original"], "HELLO 2")

    def test_preload_detect_ocr_page_writes_cache_without_mutating_project(self) -> None:
        from editor_vision_cache import build_detect_ocr_cache_key, read_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "idioma_origem": "en",
                "contexto": {},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [],
                        "textos": [],
                    }
                ],
            }
            original_project_json = json.dumps(project, ensure_ascii=False)
            project_path.write_text(original_project_json, encoding="utf-8")

            cache_key = build_detect_ocr_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )
            ocr_page = {
                "texts": [
                    {
                        "id": "tl_001_001",
                        "bbox": [10, 20, 110, 120],
                        "balloon_bbox": [10, 20, 110, 120],
                        "text": "HELLO",
                        "confidence": 0.93,
                        "tipo": "fala",
                    }
                ],
                "_vision_blocks": [{"bbox": [10, 20, 110, 120], "confidence": 0.88}],
            }

            with patch("ocr.detector.run_ocr", return_value=ocr_page) as run_ocr:
                with patch("ocr.contextual_reviewer.contextual_review_page", side_effect=lambda page, *_: page):
                    with patch("layout.balloon_layout.enrich_page_layout", side_effect=lambda page: page):
                        with patch("main.render_page_image") as render_page:
                            with patch("main.emit") as emit:
                                result = main._preload_detect_ocr_page(project_path, 0)

            self.assertEqual(result["cache"], "ready")
            run_ocr.assert_called_once()
            render_page.assert_not_called()
            emit.assert_not_called()
            self.assertEqual(project_path.read_text(encoding="utf-8"), original_project_json)
            cached = read_cache_entry(cache_key)
            self.assertEqual(cached["text_layers"][0]["original"], "HELLO")
            self.assertEqual(cached["inpaint_blocks"][0]["bbox"], [10, 20, 110, 120])

    def test_preload_ocr_layers_page_writes_cache_without_mutating_project(self) -> None:
        from editor_vision_cache import build_ocr_layers_cache_key, read_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            originals_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")

            layer = {
                "id": "current",
                "bbox": [10, 20, 110, 120],
                "source_bbox": [10, 20, 110, 120],
                "layout_bbox": [10, 20, 110, 120],
                "original": "",
                "translated": "",
                "tipo": "fala",
            }
            project = {
                "idioma_origem": "en",
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [layer],
                        "textos": [],
                    }
                ],
            }
            original_project_json = json.dumps(project, ensure_ascii=False)
            project_path.write_text(original_project_json, encoding="utf-8")

            cache_key = build_ocr_layers_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                layers=[layer],
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )

            with patch("ocr.detector.run_ocr_on_block", return_value=("CACHED OCR", 0.96)) as run_ocr_on_block:
                with patch("main.render_page_image") as render_page:
                    with patch("main.emit") as emit:
                        result = main._preload_ocr_layers_page(project_path, 0)

            self.assertEqual(result["cache"], "ready")
            run_ocr_on_block.assert_called_once()
            render_page.assert_not_called()
            emit.assert_not_called()
            self.assertEqual(project_path.read_text(encoding="utf-8"), original_project_json)
            cached = read_cache_entry(cache_key)
            self.assertEqual(cached["layer_updates"][0]["id"], "current")
            self.assertEqual(cached["layer_updates"][0]["original"], "CACHED OCR")
            self.assertEqual(cached["layer_updates"][0]["ocr_confidence"], 0.96)

    def test_project_inpaint_block_preserves_raw_bbox_with_text_anchor_metadata(self) -> None:
        block = main._project_inpaint_block_from_vision_block(
            {
                "bbox": [8, 12873, 749, 13652],
                "confidence": 0.91,
                "text_pixel_bbox": [303, 13497, 691, 13630],
                "line_polygons": [[[303, 13497], [691, 13497], [691, 13630], [303, 13630]]],
                "balloon_type": "textured",
            }
        )

        self.assertIsNotNone(block)
        self.assertEqual(block["bbox"], [8, 12873, 749, 13652])
        self.assertEqual(block["source_bbox"], [8, 12873, 749, 13652])
        self.assertEqual(block["text_pixel_bbox"], [303, 13497, 691, 13630])

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
        self.assertEqual(legacy["balloon_type"], "")

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

            def fake_run_ocr_on_block(image_path, bbox, **_kwargs):
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

    def test_run_ocr_page_ignores_cache_when_updates_match_no_layers(self) -> None:
        from editor_vision_cache import build_ocr_layers_cache_key, build_ocr_layers_payload, write_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")
            (translated_dir / "001.jpg").write_bytes(b"fake")

            layer = {
                "id": "current",
                "bbox": [10, 20, 110, 120],
                "source_bbox": [10, 20, 110, 120],
                "layout_bbox": [10, 20, 110, 120],
                "original": "",
                "translated": "",
                "tipo": "fala",
            }
            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [layer],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            cache_key = build_ocr_layers_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                layers=[layer],
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )
            write_cache_entry(
                cache_key,
                build_ocr_layers_payload(
                    page_index=0,
                    layer_updates=[{"id": "other", "original": "CACHED", "ocr_confidence": 0.1, "confianca_ocr": 0.1}],
                ),
            )

            calls = []

            def fake_run_ocr_on_block(image_path, bbox, **_kwargs):
                calls.append((Path(image_path), list(bbox)))
                return ("LIVE", 0.92)

            with patch("ocr.detector.run_ocr_on_block", side_effect=fake_run_ocr_on_block):
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_ocr_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            saved_layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(saved_layer["original"], "LIVE")
            self.assertEqual(saved_layer["ocr_confidence"], 0.92)
            self.assertEqual(calls, [(originals_dir / "001.jpg", [10, 20, 110, 120])])

    def test_run_ocr_page_ignores_cache_updates_with_empty_ids(self) -> None:
        from editor_vision_cache import build_ocr_layers_cache_key, build_ocr_layers_payload, write_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")
            (translated_dir / "001.jpg").write_bytes(b"fake")

            layer = {
                "id": "",
                "bbox": [10, 20, 110, 120],
                "source_bbox": [10, 20, 110, 120],
                "layout_bbox": [10, 20, 110, 120],
                "original": "",
                "translated": "",
                "tipo": "fala",
            }
            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [layer],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            cache_key = build_ocr_layers_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                layers=[layer],
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )
            write_cache_entry(
                cache_key,
                build_ocr_layers_payload(
                    page_index=0,
                    layer_updates=[
                        {"original": "NO_ID", "ocr_confidence": 0.1, "confianca_ocr": 0.1},
                        {"id": "", "original": "EMPTY", "ocr_confidence": 0.2, "confianca_ocr": 0.2},
                        {"id": "   ", "original": "BLANK", "ocr_confidence": 0.3, "confianca_ocr": 0.3},
                    ],
                ),
            )

            calls = []

            def fake_run_ocr_on_block(image_path, bbox, **_kwargs):
                calls.append((Path(image_path), list(bbox)))
                return ("LIVE_EMPTY_ID", 0.94)

            with patch("ocr.detector.run_ocr_on_block", side_effect=fake_run_ocr_on_block):
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_ocr_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            saved_layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(saved_layer["original"], "LIVE_EMPTY_ID")
            self.assertEqual(saved_layer["ocr_confidence"], 0.94)
            self.assertEqual(calls, [(originals_dir / "001.jpg", [10, 20, 110, 120])])

    def test_run_ocr_page_applies_matching_cache_without_live_ocr(self) -> None:
        from editor_vision_cache import build_ocr_layers_cache_key, build_ocr_layers_payload, write_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")
            (translated_dir / "001.jpg").write_bytes(b"fake")

            layer = {
                "id": "current",
                "bbox": [10, 20, 110, 120],
                "source_bbox": [10, 20, 110, 120],
                "layout_bbox": [10, 20, 110, 120],
                "original": "",
                "translated": "",
                "tipo": "fala",
            }
            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [layer],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            cache_key = build_ocr_layers_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                layers=[layer],
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )
            write_cache_entry(
                cache_key,
                build_ocr_layers_payload(
                    page_index=0,
                    layer_updates=[{"id": "current", "original": "CACHED", "ocr_confidence": 0.97, "confianca_ocr": 0.97}],
                ),
            )

            with patch("ocr.detector.run_ocr_on_block") as run_ocr_on_block:
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_ocr_page(project_path, 0)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            saved_layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(saved_layer["original"], "CACHED")
            self.assertEqual(saved_layer["ocr_confidence"], 0.97)
            run_ocr_on_block.assert_not_called()

    def test_run_ocr_page_region_bypasses_matching_cache(self) -> None:
        from editor_vision_cache import build_ocr_layers_cache_key, build_ocr_layers_payload, write_cache_entry

        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            originals_dir = Path(tmp) / "originals"
            translated_dir = Path(tmp) / "translated"
            originals_dir.mkdir()
            translated_dir.mkdir()
            (originals_dir / "001.jpg").write_bytes(b"fake")
            (translated_dir / "001.jpg").write_bytes(b"fake")

            layer = {
                "id": "current",
                "bbox": [10, 20, 110, 120],
                "source_bbox": [10, 20, 110, 120],
                "layout_bbox": [10, 20, 110, 120],
                "original": "",
                "translated": "",
                "tipo": "fala",
            }
            project = {
                "idioma_origem": "en",
                "engine_preset_id": "",
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_original": "originals/001.jpg",
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [layer],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            cache_key = build_ocr_layers_cache_key(
                project_path=project_path,
                page_index=0,
                image_path=originals_dir / "001.jpg",
                layers=[layer],
                idioma_origem="en",
                engine_preset_id="",
                schema_version=1,
            )
            write_cache_entry(
                cache_key,
                build_ocr_layers_payload(
                    page_index=0,
                    layer_updates=[{"id": "current", "original": "CACHED", "ocr_confidence": 0.97, "confianca_ocr": 0.97}],
                ),
            )

            calls = []

            def fake_run_ocr_on_block(image_path, bbox, **_kwargs):
                calls.append((Path(image_path), list(bbox)))
                return ("REGION_LIVE", 0.89)

            with patch("ocr.detector.run_ocr_on_block", side_effect=fake_run_ocr_on_block):
                with patch("main.render_page_image"):
                    with patch("main.emit_progress"):
                        with patch("main.emit"):
                            main._run_ocr_page(project_path, 0, region={"bbox": [0, 0, 200, 200]})

            saved = json.loads(project_path.read_text(encoding="utf-8"))
            saved_layer = saved["paginas"][0]["text_layers"][0]
            self.assertEqual(saved_layer["original"], "REGION_LIVE")
            self.assertEqual(saved_layer["ocr_confidence"], 0.89)
            self.assertEqual(calls, [(originals_dir / "001.jpg", [10, 20, 110, 120])])

    def test_page_action_options_parse_engine_preset_and_languages(self) -> None:
        region, options = main._page_action_options_from_args(
            [
                "--region-bbox",
                "10,20,110,120",
                "--source-lang",
                "ja",
                "--target-lang",
                "pt-BR",
                "--engine-preset",
                "manga",
            ]
        )
        project = {}

        main._apply_page_action_language_options(project, options)

        self.assertEqual(region["bbox"], [10, 20, 110, 120])
        self.assertEqual(project["idioma_origem"], "ja")
        self.assertEqual(project["idioma_destino"], "pt-BR")
        self.assertEqual(project["engine_preset_id"], "manga")

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

    def test_normalize_text_layer_for_renderer_preserves_trace_metadata(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_006_band_112",
                "source_trace_ids": [
                    "ocr_001@page_006_band_112",
                    "ocr_002@page_006_band_112",
                ],
                "source_text_ids": ["ocr_001", "ocr_002"],
                "merge_reason": "clustered_line_fragments",
                "ocr_merged_source_count": 2,
                "text_instance_id": "instance-112",
                "page_id": "page_006",
                "band_id": "page_006_band_112",
                "strip_band_y_top": 6400,
                "band_y_top": 120,
                "bbox": [153, 6430, 322, 6449],
                "text_pixel_bbox": [153, 6430, 322, 6449],
                "translated": "O QUE E ISSO?!",
            },
            page_number=6,
            layer_index=4,
        )

        self.assertEqual(layer["trace_id"], "ocr_001@page_006_band_112")
        self.assertEqual(
            layer["source_trace_ids"],
            ["ocr_001@page_006_band_112", "ocr_002@page_006_band_112"],
        )
        self.assertEqual(layer["source_text_ids"], ["ocr_001", "ocr_002"])
        self.assertEqual(layer["merge_reason"], "clustered_line_fragments")
        self.assertEqual(layer["ocr_merged_source_count"], 2)
        self.assertEqual(layer["text_instance_id"], "instance-112")
        self.assertEqual(layer["page_id"], "page_006")
        self.assertEqual(layer["band_id"], "page_006_band_112")
        self.assertEqual(layer["strip_band_y_top"], 6400)
        self.assertEqual(layer["band_y_top"], 120)

    def test_normalize_text_layer_for_renderer_preserves_ocr_normalization_metadata(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "ocr_001",
                "original": "What!Then,why did we come to the cafe,what are you hiding?",
                "raw_ocr": "What!Then,why did we come to the cafe,what are you hiding?",
                "normalized_ocr": "What! Then, why did we come to the cafe, what are you hiding?",
                "normalized_text_final": "What! Then, why did we come to the cafe, what are you hiding?",
                "normalization": {
                    "changed": True,
                    "rules_applied": ["repair_missing_punctuation_spacing"],
                },
                "normalization_trace": {
                    "changed": True,
                    "rules_applied": ["repair_missing_punctuation_spacing"],
                },
                "translated": "O QUE! ENTAO, POR QUE VIEMOS AO CAFE?",
                "bbox": [72, 1286, 313, 1382],
            },
            page_number=17,
            layer_index=1,
        )

        self.assertEqual(layer["raw_ocr"], "What!Then,why did we come to the cafe,what are you hiding?")
        self.assertEqual(layer["normalized_ocr"], "What! Then, why did we come to the cafe, what are you hiding?")
        self.assertEqual(layer["normalized_text_final"], "What! Then, why did we come to the cafe, what are you hiding?")
        self.assertTrue(layer["normalization"]["changed"])
        self.assertEqual(layer["normalization_trace"]["rules_applied"], ["repair_missing_punctuation_spacing"])

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

            def fake_run_ocr_on_block(image_path, bbox, **_kwargs):
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

    def test_build_text_layer_preserves_connected_balloon_metadata(self) -> None:
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

        self.assertEqual(layer.get("connected_position_bboxes"), [[113, 1513, 345, 1712], [462, 1568, 705, 1767]])
        self.assertEqual(layer.get("connected_text_groups"), [[113, 1513, 373, 1722], [432, 1558, 705, 1767]])
        self.assertEqual(layer.get("connected_detection_confidence"), 1.0)
        self.assertEqual(layer.get("layout_group_size"), 2)
        self.assertEqual(layer.get("balloon_subregions"), [[113, 1513, 395, 1767], [409, 1513, 705, 1767]])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].get("balloon_subregions"), [[113, 1513, 395, 1767], [409, 1513, 705, 1767]])
        self.assertEqual(blocks[0].get("connected_position_bboxes"), [[113, 1513, 345, 1712], [462, 1568, 705, 1767]])

    def test_build_text_layer_sanitizes_auto_style_before_project_json(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "HELLO",
                "bbox": [0, 0, 100, 100],
                "layout_bbox": [0, 0, 100, 100],
                "tipo": "fala",
                "confidence": 0.92,
                "background_rgb": [250, 250, 250],
                "estilo": {
                    "fonte": "Newrotic.ttf",
                    "cor": "#FFFFFF",
                    "contorno": "#000000",
                    "contorno_px": 2,
                    "glow": True,
                    "sombra": True,
                },
            },
            translated="OLA",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#000000")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)
        self.assertFalse(layer["estilo"]["glow"])
        self.assertFalse(layer["estilo"]["sombra"])

    def test_build_text_layer_forces_black_text_for_white_balloon(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "RIGHT?",
                "bbox": [10, 20, 140, 70],
                "tipo": "fala",
                "confidence": 0.92,
                "background_rgb": [155, 155, 155],
                "balloon_type": "white",
                "estilo": {"cor": "#FFFFFF"},
            },
            translated="CERTO?",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["estilo"]["cor"], "#000000")

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


    def test_build_text_layer_preserves_smart_skip_audit_for_project_json(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "text": "FOR FASTER UPDATE",
                "bbox": [10, 10, 180, 40],
                "skip_processing": True,
                "skip_reason": "smart_skip",
                "content_class": "url_watermark",
                "smart_skip_decision": {
                    "category": "credit_or_watermark",
                    "reason": "opening-page credit or reader/update notice",
                },
            },
            translated="FOR FASTER UPDATE",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertFalse(layer["skip_processing"])
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertEqual(layer["route_reason"], "dialogue_balloon_with_english_text")
        self.assertIsNone(layer["skip_reason"])
        self.assertEqual(layer["content_class"], "text")
        self.assertEqual(layer["translate_policy"], "translate")
        self.assertEqual(layer["render_policy"], "normal")
        self.assertEqual(layer["smart_skip_decision"]["category"], "credit_or_watermark")

    def test_build_text_layer_preserves_route_action_contract_for_project_json(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "text": "Read at ASURACOMIC.NET",
                "bbox": [10, 10, 180, 40],
                "skip_processing": False,
                "content_class": "url_watermark",
                "translate_policy": "skip_translation",
                "render_policy": "remove",
                "route_action": "inpaint_only",
                "route_reason": "watermark_detected",
                "is_watermark": True,
            },
            translated="Read at ASURACOMIC.NET",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertFalse(layer["skip_processing"])
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertEqual(layer["route_reason"], "dialogue_balloon_with_english_text")
        self.assertTrue(layer["is_watermark"])
        self.assertFalse(layer["is_non_english"])
        self.assertEqual(layer["translate_policy"], "translate")
        self.assertEqual(layer["render_policy"], "normal")

    def test_ensure_project_route_action_contract_fills_missing_route_actions(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "dialogue-1",
                            "text": "What are you doing?",
                            "content_class": "dialogue",
                            "skip_processing": False,
                        },
                        {
                            "id": "legacy-skip",
                            "text": "////",
                            "skip_processing": True,
                            "skip_reason": "legacy_noise",
                        },
                    ],
                    "texts": [
                        {
                            "id": "watermark-1",
                            "text": "Read at example.com",
                            "content_class": "url_watermark",
                            "skip_processing": True,
                        }
                    ],
                }
            ]
        }

        audit = main._ensure_project_route_action_contract(project)

        layers = project["paginas"][0]["text_layers"]
        texts = project["paginas"][0]["texts"]
        self.assertEqual(audit["filled_missing_count"], 0)
        self.assertEqual(layers[0]["route_action"], "translate_inpaint_render")
        self.assertFalse(layers[0]["skip_processing"])
        self.assertEqual(layers[1]["route_action"], "translate_inpaint_render")
        self.assertFalse(layers[1]["skip_processing"])
        self.assertEqual(texts[0]["route_action"], "translate_inpaint_render")
        self.assertFalse(texts[0]["skip_processing"])

    def test_ensure_project_route_action_contract_overrides_special_translate_routes(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "credit",
                            "content_class": "scanlator_credit",
                            "route_action": "translate_inpaint_render",
                            "route_reason": "dialogue_balloon_with_english_text",
                            "skip_processing": False,
                        }
                    ],
                    "textos": [],
                }
            ]
        }

        audit = main._ensure_project_route_action_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["overridden_special_count"], 0)
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertEqual(layer["route_reason"], "dialogue_balloon_with_english_text")
        self.assertFalse(layer["skip_processing"])

    def test_ensure_project_route_action_contract_overrides_noise_translate_routes(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "noise",
                            "content_class": "noise",
                            "route_action": "translate_inpaint_render",
                            "route_reason": "translate_inpaint_render",
                            "skip_processing": False,
                        }
                    ],
                    "textos": [],
                }
            ]
        }

        audit = main._ensure_project_route_action_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["overridden_special_count"], 0)
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertFalse(layer["skip_processing"])

    def test_ensure_project_route_action_contract_preserves_noop_name_without_glyph_evidence(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "name-noop",
                            "original": "Hosu?",
                            "translated": "Hosu?",
                            "content_class": "dialogue",
                            "route_action": "translate_inpaint_render",
                            "route_reason": "dialogue_balloon_with_english_text",
                            "skip_processing": False,
                            "mask_evidence": {
                                "kind": "none",
                                "raw_mask_pixels": 0,
                                "expanded_mask_pixels": 0,
                                "evidence_score": 0.0,
                                "fast_fill_allowed": False,
                                "fast_fill_reject_reasons": ["raw_mask_pixels_zero"],
                            },
                            "qa_flags": ["fast_fill_no_glyph_evidence"],
                        }
                    ],
                    "textos": [],
                }
            ]
        }

        audit = main._ensure_project_route_action_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["preserved_noop_without_glyph_count"], 0)
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertFalse(layer["skip_processing"])
        self.assertEqual(layer["render_policy"], "normal")
        self.assertIn("fast_fill_no_glyph_evidence", layer["qa_flags"])

    def test_ensure_project_route_action_contract_preserves_noop_name_before_mask_flags(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "name-noop",
                            "original": "Hosu?",
                            "translated": "Hosu?",
                            "content_class": "dialogue",
                            "route_action": "translate_inpaint_render",
                            "route_reason": "dialogue_balloon_with_english_text",
                            "skip_processing": False,
                            "qa_flags": [],
                        }
                    ],
                    "textos": [],
                }
            ]
        }

        audit = main._ensure_project_route_action_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["preserved_noop_without_glyph_count"], 0)
        self.assertEqual(layer["route_action"], "translate_inpaint_render")
        self.assertEqual(layer["route_reason"], "dialogue_balloon_with_english_text")
        self.assertEqual(layer["render_policy"], "normal")

    def test_filter_render_plan_qa_flags_drops_resolved_fit_flag(self) -> None:
        flags = {"fit_below_minimum_legible", "TEXT_CLIPPED"}
        entry = {
            "fit_status": "ok",
            "fit_attempts": [{"font_px": 13, "lines": 3, "status": "ok"}],
            "render_bbox": [20, 20, 80, 50],
            "safe_text_box": [10, 10, 90, 60],
            "target_bbox": [0, 0, 100, 80],
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertEqual(filtered, set())

    def test_filter_render_plan_qa_flags_keeps_render_fit_overflow_evidence(self) -> None:
        flags = {"TEXT_OVERFLOW", "safe_text_box_recomputed"}
        entry = {
            "fit_status": "ok",
            "render_bbox": [173, 10900, 506, 10919],
            "safe_text_box": [144, 10888, 535, 10919],
            "target_bbox": [171, 10866, 262, 10905],
            "balloon_bbox": [3, 10860, 651, 10986],
            "qa_metrics": {
                "render_balloon_containment": 0.3363,
                "render_fit": {
                    "flags": ["TEXT_OVERFLOW"],
                    "render_bbox": [173, 10900, 506, 10919],
                    "safe_text_box": [144, 10888, 535, 10919],
                    "target_bbox": [3, 10860, 285, 10986],
                    "balloon_bbox": [3, 10860, 285, 10986],
                },
            },
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertIn("TEXT_OVERFLOW", filtered)
        self.assertIn("safe_text_box_recomputed", filtered)

    def test_filter_render_plan_qa_flags_drops_stale_tiny_render_fit_evidence(self) -> None:
        flags = {"TEXT_CLIPPED", "TEXT_OVERFLOW", "weak_text_residual_after_inpaint"}
        entry = {
            "fit_status": "ok",
            "render_bbox": [200, 3420, 661, 3902],
            "safe_text_box": [200, 3455, 661, 3907],
            "target_bbox": [0, 3402, 800, 4039],
            "balloon_bbox": [0, 3402, 800, 4039],
            "qa_metrics": {
                "render_balloon_containment": 0.8814,
                "render_fit": {
                    "flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW"],
                    "render_bbox": [200, 3420, 318, 3458],
                    "safe_text_box": [200, 3455, 370, 3464],
                    "target_bbox": [134, 3415, 304, 3464],
                    "balloon_bbox": [134, 3415, 304, 3464],
                },
            },
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertEqual(filtered, {"weak_text_residual_after_inpaint"})

    def test_filter_render_plan_qa_flags_drops_resolved_missing_render_flag(self) -> None:
        flags = {"missing_render_bbox", "safe_text_box_recomputed"}
        entry = {
            "fit_status": "ok",
            "render_bbox": [20, 20, 80, 50],
            "safe_text_box": [10, 10, 90, 60],
            "target_bbox": [0, 0, 100, 80],
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertEqual(filtered, {"safe_text_box_recomputed"})

    def test_hydrate_project_render_metadata_uses_candidates_before_render_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_024_band_064",
                        "text_id": "ocr_001",
                        "band_id": "page_024_band_064",
                        "page_id": "page_024",
                        "target_bbox": [75, 96, 248, 396],
                        "render_bbox": [97, 186, 226, 306],
                        "safe_text_box": [94, 135, 228, 357],
                        "fit_status": "ok",
                        "qa_flags": ["safe_text_box_recomputed"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_024_band_064",
                        "text_id": "ocr_001",
                        "band_id": "page_024_band_064",
                        "page_id": "page_024",
                        "bboxes": {
                            "bbox": {"value": [75, 1180, 248, 1480]},
                            "text_pixel_bbox": {"value": [86, 1189, 241, 1482]},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_024_band_064",
                                "band_id": "page_024_band_064",
                                "page_id": "page_024",
                                "route_action": "translate_inpaint_render",
                                "translated": "Quando estou atualizando meu equipamento...",
                                "bbox": [86, 1189, 241, 1482],
                                "text_pixel_bbox": [86, 1189, 241, 1482],
                                "balloon_bbox": [75, 1180, 248, 1480],
                                "qa_flags": [],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["target_bbox"], [75, 1180, 248, 1480])
        self.assertEqual(layer["render_bbox"], [97, 1270, 226, 1390])
        self.assertEqual(layer["safe_text_box"], [94, 1219, 228, 1441])
        self.assertEqual(layer["fit_status"], "ok")
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_ensure_project_render_contract_drops_stale_fit_flag_when_fit_is_ok(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "dialogue",
                            "route_action": "translate_inpaint_render",
                            "skip_processing": False,
                            "translated": "Claro, fique seguro!",
                            "render_bbox": [466, 1734, 574, 1768],
                            "safe_text_box": [462, 1724, 577, 1778],
                            "fit_status": "ok",
                            "fit_attempts": [{"font_px": 12, "lines": 2, "status": "ok"}],
                            "qa_flags": ["fit_below_minimum_legible", "safe_text_box_recomputed"],
                        }
                    ],
                }
            ]
        }

        audit = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["dropped_stale_fit_flag_count"], 1)
        self.assertEqual(layer["fit_status"], "ok")
        self.assertEqual(layer["qa_flags"], ["safe_text_box_recomputed"])

    def test_ensure_project_render_contract_normalizes_stale_fit_status_when_attempts_are_ok(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "dialogue",
                            "route_action": "translate_inpaint_render",
                            "skip_processing": False,
                            "translated": "Uau... pensei que tinha voltado para o exército..",
                            "render_bbox": [306, 249, 577, 348],
                            "safe_text_box": [306, 222, 582, 359],
                            "fit_status": "below_minimum_legible",
                            "fit_attempts": [
                                {"font_px": 12, "lines": 4, "status": "ok"},
                                {"font_px": 12, "lines": 2, "status": "ok"},
                            ],
                            "qa_flags": ["fit_below_minimum_legible"],
                        }
                    ],
                }
            ]
        }

        audit = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["normalized_fit_status_count"], 1)
        self.assertEqual(audit["dropped_stale_fit_flag_count"], 1)
        self.assertEqual(layer["fit_status"], "ok")
        self.assertEqual(layer["qa_flags"], [])

    def test_ensure_project_render_contract_keeps_low_luma_render_on_art_for_non_white_layer(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "dialogue",
                            "route_action": "translate_inpaint_render",
                            "skip_processing": False,
                            "translated": "Esta feito... vamos mudar!",
                            "balloon_type": "textured",
                            "render_bbox": [276, 5227, 524, 5244],
                            "safe_text_box": [88, 5034, 712, 5438],
                            "fit_status": "ok",
                            "fit_attempts": [{"font_px": 19, "lines": 1, "status": "ok"}],
                            "qa_flags": ["render_on_art_suspected", "safe_text_box_recomputed"],
                            "qa_metrics": {"render_background_luma": 132.7},
                        }
                    ],
                }
            ]
        }

        audit = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["dropped_stale_render_background_flag_count"], 0)
        self.assertEqual(layer["qa_flags"], ["render_on_art_suspected", "safe_text_box_recomputed"])

    def test_filter_render_plan_qa_flags_drops_stale_render_on_art_when_background_is_white(self) -> None:
        flags = {"render_on_art_suspected", "safe_text_box_recomputed"}
        entry = {
            "balloon_type": "white",
            "qa_metrics": {"render_background_luma": 255.0},
            "render_bbox": [463, 3541, 706, 3579],
            "safe_text_box": [451, 3526, 706, 3579],
            "target_bbox": [212, 3325, 706, 3613],
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertEqual(filtered, {"safe_text_box_recomputed"})

    def test_filter_render_plan_qa_flags_keeps_render_on_art_when_background_is_dark(self) -> None:
        flags = {"render_on_art_suspected", "safe_text_box_recomputed"}
        entry = {
            "balloon_type": "textured",
            "qa_metrics": {"render_background_luma": 132.7},
            "render_bbox": [276, 5227, 524, 5244],
            "safe_text_box": [88, 5034, 712, 5438],
            "target_bbox": [252, 5018, 322, 5029],
        }

        filtered = main._filter_render_plan_qa_flags(entry, flags)

        self.assertIn("render_on_art_suspected", filtered)
        self.assertIn("safe_text_box_recomputed", filtered)

    def test_debug_qa_propagation_does_not_readd_resolved_fit_flag(self) -> None:
        project = {
            "_work_dir": "dummy",
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "trace_id": "ocr_001@page_001_band_001",
                            "route_action": "translate_inpaint_render",
                            "skip_processing": False,
                            "render_bbox": [20, 20, 80, 50],
                            "safe_text_box": [10, 10, 90, 60],
                            "fit_status": "ok",
                            "qa_flags": [],
                        }
                    ]
                }
            ],
        }
        claim = {
            "identity_groups": [["ocr_001@page_001_band_001", "ocr_001"]],
            "flags": {"fit_below_minimum_legible"},
            "source": "render_plan",
        }

        with patch.object(main, "_debug_root_from_project", return_value=Path("dummy-debug")):
            with patch.object(main, "_collect_render_plan_qa_flags", return_value=[claim]):
                with patch.object(main, "_collect_mask_decision_qa_flags", return_value=[]):
                    with patch.object(main, "_collect_inpaint_decision_qa_flags", return_value=[]):
                        audit = main._propagate_debug_qa_flags_to_project(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(layer["qa_flags"], [])
        self.assertEqual(audit["summary"]["project_layer_flags"], 0)

    def test_final_page_space_text_layers_drop_stale_safe_box_before_rerender(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "sign",
                        "original": "RIGHT TURN ONLY",
                        "translated": "APENAS VIRAR À DIREITA",
                        "bbox": [370, 7567, 435, 7640],
                        "source_bbox": [370, 7567, 435, 7640],
                        "balloon_bbox": [370, 7567, 435, 7640],
                        "safe_text_box": [387, 7584, 418, 7623],
                        "render_bbox": [391, 7585, 413, 7621],
                        "fit_status": "below_minimum_legible",
                        "qa_flags": ["fit_below_minimum_legible"],
                        "route_action": "translate_inpaint_render",
                    }
                ]
            },
            page_number=1,
        )

        self.assertEqual(len(normalized), 1)
        self.assertNotIn("safe_text_box", normalized[0])
        self.assertNotIn("fit_status", normalized[0])
        self.assertEqual(normalized[0]["qa_flags"], ["fit_below_minimum_legible"])

    def test_final_page_space_text_layers_preserve_mask_evidence(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "dialogue",
                        "original": "Hello",
                        "translated": "Olá",
                        "bbox": [10, 10, 90, 40],
                        "balloon_bbox": [0, 0, 100, 60],
                        "route_action": "translate_inpaint_render",
                        "mask_evidence": {
                            "kind": "glyph_segmentation",
                            "raw_mask_pixels": 120,
                            "expanded_mask_pixels": 160,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                    }
                ]
            },
            page_number=1,
        )

        self.assertEqual(normalized[0]["mask_evidence"]["raw_mask_pixels"], 120)

    def test_final_page_space_text_layers_preserve_bubble_geometry(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "dialogue",
                        "original": "Ajussi How long",
                        "translated": "Ajussi quanto tempo",
                        "bbox": [135, 4337, 274, 4439],
                        "source_bbox": [125, 4294, 656, 4853],
                        "text_pixel_bbox": [135, 4337, 274, 4439],
                        "balloon_bbox": [214, 4325, 269, 4378],
                        "bubble_mask_bbox": [125, 4294, 656, 4853],
                        "bubble_inner_bbox": [162, 4331, 619, 4816],
                        "source_trace_ids": [
                            "ocr_001@page_003_band_049",
                            "ocr_002@page_003_band_049",
                        ],
                        "route_action": "translate_inpaint_render",
                    }
                ]
            },
            page_number=3,
        )

        self.assertEqual(normalized[0]["bubble_mask_bbox"], [125, 4294, 656, 4853])
        self.assertEqual(normalized[0]["bubble_inner_bbox"], [162, 4331, 619, 4816])
        self.assertEqual(
            normalized[0]["source_trace_ids"],
            ["ocr_001@page_003_band_049", "ocr_002@page_003_band_049"],
        )

    def test_normalize_text_layer_for_renderer_preserves_content_class(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "sign-1",
                "original": "TEXT: DARLING KARAOKE",
                "translated": "TEXTO: QUERIDO KARAOKE",
                "bbox": [45, 9305, 205, 9322],
                "source_bbox": [45, 9305, 205, 9322],
                "text_pixel_bbox": [118, 9311, 145, 9316],
                "balloon_bbox": [10, 9293, 240, 9334],
                "tipo": "narracao",
                "balloon_type": "white",
                "content_class": "sign",
            },
            3,
            12,
        )

        self.assertEqual(layer["content_class"], "sign")

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
                    "content_class": "dialogue",
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
        self.assertEqual(page["textos"][0]["content_class"], "text")


if __name__ == "__main__":
    unittest.main()
