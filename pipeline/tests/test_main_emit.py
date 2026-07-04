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

import cv2
import numpy as np

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

    def test_normalize_dark_panel_contract_preserves_connected_dark_bubble(self) -> None:
        layer = {
            "id": "ocr_001",
            "bubble_mask_source": "image_dark_bubble_mask",
            "bubbleMaskSource": "image_dark_bubble_mask",
            "layout_profile": "connected_balloon",
            "block_profile": "dark_bubble",
            "balloon_bbox": [83, 32, 744, 691],
            "bubble_mask_bbox": [83, 32, 744, 691],
            "balloon_subregions": [[83, 32, 385, 691], [385, 32, 744, 691]],
            "connected_lobe_bboxes": [[83, 32, 385, 691], [385, 32, 744, 691]],
            "connected_balloon_orientation": "left-right",
            "qa_flags": [
                "connected_layout_disabled_dark_panel_visual_mask",
                "dark_connected_lobes_repaired_from_visual_mask",
                "dark_bubble_connected_lobe_passthrough",
            ],
            "qa_metrics": {
                "image_dark_bubble_mask": {
                    "source": "image_dark_bubble_mask",
                    "mask_bbox": [83, 32, 744, 691],
                }
            },
        }

        normalized = main._normalize_dark_panel_rect_contract_layer(layer)

        self.assertEqual(normalized["bubble_mask_source"], "image_dark_bubble_mask")
        self.assertEqual(normalized["bubbleMaskSource"], "image_dark_bubble_mask")
        self.assertEqual(normalized["layout_profile"], "connected_balloon")
        self.assertEqual(normalized["balloon_bbox"], [83, 32, 744, 691])
        self.assertIn("image_dark_bubble_mask", normalized.get("qa_metrics") or {})

    def test_rehome_cross_page_band_layer_moves_merged_text_to_matching_page(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "visible": True,
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                            "traduzido": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                            "safe_text_box": [278, 16257, 545, 16383],
                            "render_bbox": [278, 16257, 545, 16383],
                            "target_bbox": [234, 16180, 575, 16368],
                        }
                    ],
                },
                {
                    "numero": 3,
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "text_id": "ocr_002",
                            "trace_id": "ocr_002@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "visible": True,
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "traduzido": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "safe_text_box": [275, 16138, 547, 16321],
                            "render_bbox": [275, 16138, 547, 16321],
                            "target_bbox": [242, 16087, 580, 16372],
                            "qa_flags": ["same_balloon_fragment_merged"],
                            "source_text_ids": ["ocr_001", "ocr_002"],
                        }
                    ],
                },
            ]
        }

        moved = main._rehome_cross_page_band_layers(project)

        self.assertEqual(moved, 1)
        correct = project["paginas"][0]["text_layers"][0]
        misplaced = project["paginas"][1]["text_layers"][0]
        self.assertEqual(correct["translated"], "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...")
        self.assertEqual(correct["traduzido"], "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...")
        self.assertEqual(correct["target_bbox"], [242, 16087, 580, 16372])
        self.assertEqual(correct["safe_text_box"], [275, 16138, 547, 16321])
        self.assertEqual(correct["render_bbox"], [275, 16138, 547, 16321])
        self.assertIn("cross_page_band_rehomed", correct["qa_flags"])
        self.assertFalse(misplaced["visible"])
        self.assertEqual(misplaced["render_policy"], "merged_into_primary")

    def test_rehome_cross_page_band_layer_can_use_hidden_layer_geometry(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "visible": True,
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "traduzido": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "safe_text_box": [278, 16257, 545, 16383],
                            "render_bbox": [278, 16257, 545, 16383],
                            "target_bbox": [234, 16180, 575, 16368],
                        }
                    ],
                },
                {
                    "numero": 3,
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "text_id": "ocr_002",
                            "trace_id": "ocr_002@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "visible": False,
                            "render_policy": "merged_into_primary",
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "traduzido": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "safe_text_box": [275, 16138, 547, 16321],
                            "render_bbox": [275, 16138, 547, 16321],
                            "target_bbox": [242, 16087, 580, 16372],
                            "qa_flags": ["same_balloon_fragment_merged", "cross_page_band_rehomed"],
                        }
                    ],
                },
            ]
        }

        moved = main._rehome_cross_page_band_layers(project)

        self.assertEqual(moved, 1)
        correct = project["paginas"][0]["text_layers"][0]
        self.assertEqual(correct["target_bbox"], [242, 16087, 580, 16372])
        self.assertEqual(correct["safe_text_box"], [275, 16138, 547, 16321])
        self.assertEqual(correct["render_bbox"], [275, 16138, 547, 16321])
        self.assertTrue(correct["_cross_page_band_rehomed_geometry"])

    def test_scrub_project_local_auxiliary_bboxes_removes_band_local_render_plan_fields(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 4,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "target_bbox": [0, 4383, 425, 4714],
                            "safe_text_box": [89, 4476, 329, 4626],
                            "render_bbox": [89, 4476, 329, 4626],
                            "layout_safe_bbox": [89, 172, 329, 322],
                            "position_bbox": [88, 180, 330, 320],
                            "capacity_bbox": [88, 180, 330, 320],
                            "_safe_text_box_unclamped": [89, -59, 329, 322],
                            "bubble_inner_bbox": [89, 4476, 329, 4626],
                        }
                    ],
                }
            ]
        }

        count = main._scrub_project_local_auxiliary_bboxes(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(count, 4)
        self.assertNotIn("layout_safe_bbox", layer)
        self.assertNotIn("position_bbox", layer)
        self.assertNotIn("capacity_bbox", layer)
        self.assertNotIn("_safe_text_box_unclamped", layer)
        self.assertEqual(layer["safe_text_box"], [89, 4476, 329, 4626])
        self.assertEqual(layer["bubble_inner_bbox"], [89, 4476, 329, 4626])

    def test_merged_render_candidate_does_not_hide_distinct_bubble_layers(self) -> None:
        top_layer = {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "bbox": [148, 7248, 310, 7268],
            "text_pixel_bbox": [148, 7248, 310, 7268],
            "bubble_mask_bbox": [29, 7109, 642, 7722],
            "balloon_bbox": [0, 7013, 800, 7967],
        }
        lower_layer = {
            "id": "ocr_003",
            "text_id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "bbox": [344, 7702, 540, 7761],
            "text_pixel_bbox": [344, 7702, 540, 7761],
            "bubble_mask_bbox": [276, 7612, 598, 7799],
            "balloon_bbox": [276, 7612, 598, 7799],
        }
        candidate = {
            "text_id": "ocr_003",
            "band_id": "page_003_band_035",
            "source_text_ids": ["ocr_001", "ocr_003"],
            "source_trace_ids": ["ocr_001@page_003_band_035", "ocr_003@page_003_band_035"],
            "target_bbox": [29, 7109, 642, 7799],
            "safe_text_box": [98, 7229, 574, 7674],
            "render_bbox": [190, 7428, 480, 7474],
        }
        layers_by_identity = main._layers_by_debug_identity([top_layer, lower_layer])

        primary = main._primary_layer_for_merged_candidate(lower_layer, candidate, layers_by_identity)

        self.assertIsNone(primary)

    def test_multisource_render_candidate_does_not_overwrite_indirect_layer_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "source_text_ids": ["ocr_001", "ocr_003"],
                        "source_trace_ids": ["ocr_001@page_003_band_035", "ocr_003@page_003_band_035"],
                        "page_id": "page_003",
                        "band_id": "page_003_band_035",
                        "translated": "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?",
                        "target_bbox": [29, 96, 642, 786],
                        "bbox": [344, 652, 540, 748],
                        "layout_bbox": [276, 599, 598, 786],
                        "text_pixel_bbox": [344, 652, 540, 748],
                        "safe_text_box": [98, 216, 574, 661],
                        "render_bbox": [190, 415, 480, 461],
                        "balloon_bbox": [276, 599, 598, 786],
                        "fit_status": "ok",
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
                                "trace_id": "ocr_001@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "bbox": [148, 235, 310, 255],
                                "text_pixel_bbox": [148, 235, 310, 255],
                                "balloon_bbox": [29, 96, 642, 709],
                                "bubble_mask_bbox": [29, 96, 642, 709],
                            },
                            {
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                                "bbox": [344, 689, 540, 748],
                                "text_pixel_bbox": [344, 689, 540, 748],
                                "balloon_bbox": [276, 599, 598, 786],
                                "bubble_mask_bbox": [276, 599, 598, 786],
                            },
                        ],
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        lower_layer = project["paginas"][0]["text_layers"][1]
        self.assertEqual(audit["hydrated_layers"], 2)
        self.assertEqual(lower_layer["translated"], "QUEM ESTÁ PAGANDO HOJE?")
        self.assertNotIn("ESTOU MORRENDO", lower_layer["translated"])
        self.assertEqual(lower_layer.get("source_trace_ids"), None)

    def test_suppressed_low_containment_fragment_does_not_merge_into_neighbor_balloon(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_003_band_035",
                            "band_id": "page_003_band_035",
                            "translated": "EI, VAMOS!",
                            "bbox": [148, 7248, 310, 7268],
                            "source_bbox": [148, 7248, 310, 7268],
                            "text_pixel_bbox": [148, 7248, 310, 7268],
                            "balloon_bbox": [133, 7192, 535, 7695],
                            "bubble_mask_bbox": [133, 7192, 535, 7695],
                            "route_action": "translate_inpaint_render",
                        },
                        {
                            "id": "ocr_003",
                            "text_id": "ocr_003",
                            "trace_id": "ocr_003@page_003_band_035",
                            "band_id": "page_003_band_035",
                            "translated": "QUEM ESTÃ PAGANDO HOJE?",
                            "bbox": [344, 7702, 540, 7761],
                            "source_bbox": [344, 7702, 540, 7761],
                            "text_pixel_bbox": [344, 7702, 540, 7761],
                            "balloon_bbox": [276, 7612, 598, 7799],
                            "bubble_mask_bbox": [276, 7612, 598, 7799],
                            "route_action": "translate_inpaint_render",
                        },
                        {
                            "id": "ocr_001_fragment_2",
                            "text_id": "ocr_001_fragment_2",
                            "trace_id": "ocr_001@page_003_band_035#fragment_2",
                            "band_id": "page_003_band_035",
                            "translated": "ESTOU MORRENDO DE FOME",
                            "bbox": [345, 7665, 537, 7686],
                            "source_bbox": [345, 7665, 537, 7686],
                            "text_pixel_bbox": [345, 7665, 537, 7686],
                            "balloon_bbox": [29, 7109, 642, 7722],
                            "target_bbox": [29, 7109, 642, 7722],
                            "safe_text_box": [98, 7214, 574, 7617],
                            "render_bbox": [193, 7408, 477, 7423],
                            "route_action": "translate_inpaint_render",
                            "qa_flags": ["render_suppressed_low_containment_fragment"],
                        },
                    ]
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        lower_layer = project["paginas"][0]["text_layers"][1]
        fragment = project["paginas"][0]["text_layers"][2]
        self.assertEqual(merged, 0)
        self.assertEqual(lower_layer["translated"], "QUEM ESTÃ PAGANDO HOJE?")
        self.assertNotIn("ESTOU MORRENDO", lower_layer["translated"])
        self.assertEqual(fragment["route_action"], "translate_inpaint_render")

    def test_split_lobe_payload_repair_uses_matching_coordinate_space(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_003_band_035",
                            "band_id": "page_003_band_035",
                            "translated": "QUEM ESTÁ PAGANDO HOJE?\nESTOU MORRENDO DE FOME",
                            "bbox": [148, 10878, 537, 11316],
                            "text_pixel_bbox": [148, 10878, 537, 11316],
                            "target_bbox": [133, 10815, 327, 10957],
                            "route_action": "translate_inpaint_render",
                            "visible": True,
                            "source_trace_ids": ["ocr_003@page_003_band_035", "ocr_001@page_003_band_035"],
                            "qa_flags": ["same_balloon_fragment_merged"],
                        },
                        {
                            "id": "ocr_003",
                            "text_id": "ocr_003",
                            "trace_id": "ocr_003@page_003_band_035",
                            "band_id": "page_003_band_035",
                            "translated": "QUEM ESTÁ PAGANDO HOJE?",
                            "bbox": [344, 11332, 540, 11391],
                            "text_pixel_bbox": [344, 11332, 540, 11391],
                            "target_bbox": [245, 11197, 629, 11474],
                            "route_action": "translate_inpaint_render",
                            "visible": True,
                        },
                        {
                            "id": "ocr_001_fragment_2",
                            "text_id": "ocr_001_fragment_2",
                            "trace_id": "ocr_001@page_003_band_035#fragment_2",
                            "band_id": "page_003_band_035",
                            "translated": "ESTOU MORRENDO DE FOME",
                            "bbox": [345, 652, 537, 673],
                            "text_pixel_bbox": [345, 652, 537, 673],
                            "target_bbox": [345, 652, 537, 673],
                            "route_action": "translate_inpaint_render",
                            "visible": True,
                        },
                        {
                            "id": "ocr_001_fragment_3",
                            "text_id": "ocr_001_fragment_3",
                            "trace_id": "ocr_001@page_003_band_035#fragment_3",
                            "band_id": "page_003_band_035",
                            "translated": "EI, VAMOS!",
                            "bbox": [148, 10878, 310, 10898],
                            "text_pixel_bbox": [148, 10878, 310, 10898],
                            "target_bbox": [133, 10815, 327, 10957],
                            "route_action": "translate_inpaint_render",
                            "visible": True,
                        },
                    ]
                }
            ]
        }
        candidates = [
            {
                "trace_id": "ocr_001@page_003_band_035",
                "text_id": "ocr_001",
                "band_id": "page_003_band_035",
                "coordinate_space": "band",
                "translated": "EI, VAMOS!",
                "bbox": [148, 235, 310, 255],
                "render_bbox": [152, 235, 304, 260],
                "safe_text_box": [85, 219, 393, 271],
            },
            {
                "trace_id": "ocr_001@page_003_band_035",
                "text_id": "ocr_001",
                "band_id": "page_003_band_035",
                "coordinate_space": "band",
                "translated": "ESTOU MORRENDO DE FOME",
                "bbox": [345, 652, 537, 673],
                "render_bbox": [270, 616, 537, 673],
                "safe_text_box": [177, 590, 537, 673],
            },
            {
                "trace_id": "ocr_003@page_003_band_035",
                "text_id": "ocr_003",
                "band_id": "page_003_band_035",
                "coordinate_space": "band",
                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                "bbox": [344, 689, 540, 748],
                "render_bbox": [361, 673, 488, 743],
                "safe_text_box": [343, 662, 506, 755],
            },
            {
                "trace_id": "ocr_001@page_003_band_035",
                "text_id": "ocr_001",
                "band_id": "page_003_band_035",
                "coordinate_space": "page",
                "translated": "EI, VAMOS!",
                "bbox": [148, 10878, 310, 10898],
                "render_bbox": [152, 10878, 304, 10903],
                "safe_text_box": [85, 10862, 393, 10914],
            },
            {
                "trace_id": "ocr_001@page_003_band_035",
                "text_id": "ocr_001",
                "band_id": "page_003_band_035",
                "coordinate_space": "page",
                "translated": "ESTOU MORRENDO DE FOME",
                "bbox": [345, 11295, 537, 11316],
                "render_bbox": [270, 11259, 537, 11316],
                "safe_text_box": [177, 11233, 537, 11316],
            },
            {
                "trace_id": "ocr_003@page_003_band_035",
                "text_id": "ocr_003",
                "band_id": "page_003_band_035",
                "coordinate_space": "page",
                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                "bbox": [344, 11332, 540, 11391],
                "render_bbox": [361, 11316, 488, 11386],
                "safe_text_box": [343, 11305, 506, 11398],
            },
        ]

        repaired = main._repair_project_split_lobe_text_payloads(
            list(main._iter_project_text_layers(project)),
            candidates,
        )

        top, lower = project["paginas"][0]["text_layers"][:2]
        self.assertEqual(repaired, 2)
        self.assertEqual(top["translated"], "EI, VAMOS!")
        self.assertEqual(top["source_trace_ids"], ["ocr_001@page_003_band_035"])
        self.assertEqual(lower["translated"], "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?")
        self.assertEqual(lower["source_trace_ids"], ["ocr_001@page_003_band_035", "ocr_003@page_003_band_035"])
        self.assertEqual(main._merge_same_balloon_fragment_layers(project), 0)
        lower["visible"] = False
        lower["route_action"] = "merged_into_primary"
        lower["merged_into_trace_id"] = top["trace_id"]
        self.assertEqual(main._restore_hidden_distinct_nonfragment_layers(project), 1)
        self.assertTrue(lower["visible"])
        self.assertEqual(lower["route_action"], "translate_inpaint_render")
        self.assertEqual(main._suppress_same_identity_merged_fragments(project), 2)
        self.assertTrue(top["visible"])
        self.assertTrue(lower["visible"])
        self.assertFalse(project["paginas"][0]["text_layers"][2]["visible"])
        self.assertFalse(project["paginas"][0]["text_layers"][3]["visible"])

    def test_real_bubble_safe_area_repair_preserves_compact_fit_for_short_split_text(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_003_band_035",
                            "band_id": "page_003_band_035",
                            "translated": "EI, VAMOS!",
                            "bubble_mask_source": "image_contour_bubble_mask",
                            "bubble_mask_bbox": [133, 10815, 327, 10957],
                            "balloon_bbox": [133, 10815, 327, 10957],
                            "target_bbox": [133, 10815, 327, 10957],
                            "safe_text_box": [85, 10862, 393, 10914],
                            "render_bbox": [152, 10878, 304, 10903],
                            "fit_status": "ok",
                            "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
                            "qa_metrics": {"render_balloon_containment": 1.0},
                        }
                    ]
                }
            ]
        }

        repaired = main._repair_project_real_bubble_body_safe_areas(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(repaired["safe_area_repaired_count"], 0)
        self.assertEqual(layer["render_bbox"], [152, 10878, 304, 10903])
        self.assertEqual(layer["safe_text_box"], [85, 10862, 393, 10914])

    def test_hydrate_ignores_suppressed_low_containment_fragment_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            raw_rows = [
                {
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "translated": "EI, VAMOS!",
                    "render_bbox": [148, 235, 300, 260],
                    "safe_text_box": [85, 219, 393, 271],
                    "text_pixel_bbox": [148, 235, 310, 255],
                    "bbox": [148, 235, 310, 255],
                    "target_bbox": [29, 96, 642, 709],
                    "fit_status": "ok",
                },
                {
                    "text_id": "ocr_001_fragment_2",
                    "trace_id": "ocr_001@page_003_band_035#fragment_2",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "translated": "ESTOU MORRENDO DE FOME",
                    "render_bbox": [193, 395, 477, 410],
                    "safe_text_box": [98, 201, 574, 604],
                    "text_pixel_bbox": [345, 652, 537, 673],
                    "bbox": [345, 652, 537, 673],
                    "target_bbox": [29, 96, 642, 709],
                    "fit_status": "ok",
                    "qa_flags": ["render_suppressed_low_containment_fragment"],
                    "render_policy": "suppressed_low_containment_fragment",
                },
                {
                    "text_id": "ocr_003",
                    "trace_id": "ocr_003@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "translated": "QUEM ESTÁ PAGANDO HOJE?",
                    "render_bbox": [345, 652, 537, 673],
                    "safe_text_box": [318, 615, 610, 704],
                    "text_pixel_bbox": [345, 652, 537, 673],
                    "bbox": [345, 652, 537, 673],
                    "target_bbox": [320, 600, 640, 720],
                    "fit_status": "ok",
                },
            ]
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            (typeset_dir / "render_plan_final.jsonl").write_text(
                json.dumps(
                    {
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "page_id": "page_003",
                        "band_id": "page_003_band_035",
                        "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                        "render_bbox": [174, 409, 494, 729],
                        "safe_text_box": [174, 409, 494, 729],
                        "text_pixel_bbox": [148, 235, 310, 255],
                        "bbox": [148, 235, 310, 255],
                        "target_bbox": [29, 96, 642, 709],
                        "fit_status": "ok",
                        "qa_flags": ["safe_text_box_recomputed"],
                    },
                    ensure_ascii=False,
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
                                "trace_id": "ocr_001@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "bbox": [148, 235, 310, 255],
                                "text_pixel_bbox": [148, 235, 310, 255],
                                "balloon_bbox": [29, 96, 642, 709],
                                "render_bbox": [174, 409, 494, 729],
                                "safe_text_box": [174, 409, 494, 729],
                                "qa_flags": ["safe_text_box_recomputed", "render_suppressed_low_containment_fragment"],
                            },
                            {
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "translated": "QUEM ESTÁ PAGANDO HOJE?",
                                "bbox": [345, 652, 537, 673],
                                "text_pixel_bbox": [345, 652, 537, 673],
                                "balloon_bbox": [320, 600, 640, 720],
                            },
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layers = project["paginas"][0]["text_layers"]
        self.assertEqual(audit["suppressed_candidate_count"], 1)
        self.assertEqual(audit["restored_missing_candidate_layers"], 0)
        self.assertEqual([layer["id"] for layer in layers], ["ocr_001", "ocr_003"])
        self.assertEqual(layers[0]["translated"], "EI, VAMOS!")
        self.assertEqual(layers[1]["translated"], "QUEM ESTÁ PAGANDO HOJE?")
        self.assertEqual(layers[0]["render_bbox"], [148, 235, 300, 260])
        self.assertNotIn("ESTOU MORRENDO", layers[0]["translated"])
        self.assertNotIn("ESTOU MORRENDO", layers[1]["translated"])
        self.assertNotIn("render_suppressed_low_containment_fragment", layers[0].get("qa_flags") or [])
        self.assertNotIn("render_suppressed_low_containment_fragment", layers[1].get("qa_flags") or [])

    def test_hydrate_replaces_same_identity_suppressed_fragment_text_in_clean_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            raw_rows = [
                {
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "translated": "EI, VAMOS!",
                    "render_bbox": [148, 235, 300, 260],
                    "safe_text_box": [85, 219, 393, 271],
                    "text_pixel_bbox": [148, 235, 310, 255],
                    "bbox": [148, 235, 310, 255],
                    "target_bbox": [29, 96, 642, 709],
                    "fit_status": "ok",
                    "qa_flags": ["safe_text_box_recomputed"],
                },
                {
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "translated": "ESTOU MORRENDO DE FOME",
                    "render_bbox": [193, 395, 477, 410],
                    "safe_text_box": [98, 201, 574, 604],
                    "text_pixel_bbox": [345, 652, 537, 673],
                    "bbox": [345, 652, 537, 673],
                    "target_bbox": [29, 96, 642, 709],
                    "fit_status": "ok",
                    "qa_flags": ["safe_text_box_recomputed", "render_suppressed_low_containment_fragment"],
                },
            ]
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
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
                                "trace_id": "ocr_001@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "bbox": [148, 235, 310, 255],
                                "text_pixel_bbox": [148, 235, 310, 255],
                                "balloon_bbox": [29, 96, 642, 709],
                                "render_bbox": [174, 409, 494, 729],
                                "safe_text_box": [174, 409, 494, 729],
                                "qa_flags": ["safe_text_box_recomputed"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["suppressed_candidate_count"], 1)
        self.assertEqual(audit["low_containment_text_payload_repairs"], 1)
        self.assertEqual(layer["translated"], "EI, VAMOS!")
        self.assertEqual(layer["render_bbox"], [148, 235, 300, 260])
        self.assertNotIn("ESTOU MORRENDO", layer["translated"])
        self.assertNotIn("render_suppressed_low_containment_fragment", layer.get("qa_flags") or [])

    def test_save_project_json_syncs_legacy_textos_after_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            project = {
                "paginas": [
                    {
                        "image_layers": {"rendered": {"path": "translated/003.jpg"}},
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "translated": "EI, VAMOS!",
                                "traduzido": "EI, VAMOS!",
                                "render_bbox": [148, 235, 300, 260],
                                "safe_text_box": [85, 219, 393, 271],
                                "bbox": [148, 235, 310, 255],
                                "qa_flags": ["safe_text_box_recomputed"],
                            }
                        ],
                        "textos": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "traduzido": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "bbox": [174, 409, 494, 729],
                                "qa_flags": ["render_suppressed_low_containment_fragment"],
                            }
                        ],
                    }
                ],
            }

            main._save_project_json(project_path, project)

            saved = json.loads(project_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["paginas"][0]["text_layers"][0]["translated"], "EI, VAMOS!")
        self.assertEqual(saved["paginas"][0]["textos"][0]["translated"], "EI, VAMOS!")
        self.assertEqual(saved["paginas"][0]["textos"][0]["bbox"], [148, 235, 300, 260])
        self.assertNotIn(
            "render_suppressed_low_containment_fragment",
            saved["paginas"][0]["textos"][0].get("qa_flags") or [],
        )

    def test_hydrate_project_restores_additional_same_identity_visual_lobe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            candidates = [
                {
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "original": "HEY, LET'S GO! I'M STARVING",
                    "translated": "EI, VAMOS!",
                    "target_bbox": [29, 96, 642, 709],
                    "bbox": [148, 235, 310, 255],
                    "layout_bbox": [148, 235, 310, 255],
                    "text_pixel_bbox": [148, 235, 310, 255],
                    "safe_text_box": [85, 219, 393, 271],
                    "render_bbox": [148, 235, 300, 260],
                    "fit_status": "ok",
                },
                {
                    "text_id": "ocr_001",
                    "trace_id": "ocr_001@page_003_band_035",
                    "page_id": "page_003",
                    "band_id": "page_003_band_035",
                    "original": "HEY, LET'S GO! I'M STARVING",
                    "translated": "ESTOU MORRENDO DE FOME",
                    "target_bbox": [29, 96, 642, 709],
                    "bbox": [345, 652, 537, 673],
                    "layout_bbox": [345, 652, 537, 673],
                    "text_pixel_bbox": [345, 652, 537, 673],
                    "safe_text_box": [98, 201, 574, 604],
                    "render_bbox": [193, 395, 477, 410],
                    "fit_status": "ok",
                },
            ]
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                "".join(json.dumps(candidate) + "\n" for candidate in candidates),
                encoding="utf-8",
            )
            (typeset_dir / "render_plan_candidates.jsonl").write_text(
                json.dumps(
                    {
                        **candidates[1],
                        "text_id": "ocr_001_fragment_2",
                        "trace_id": "ocr_001@page_003_band_035#fragment_2",
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
                                "trace_id": "ocr_001@page_003_band_035",
                                "page_id": "page_003",
                                "band_id": "page_003_band_035",
                                "text": "HEY, LET'S GO! I'M STARVING",
                                "original": "HEY, LET'S GO! I'M STARVING",
                                "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                                "bbox": [148, 235, 537, 673],
                                "text_pixel_bbox": [148, 235, 537, 673],
                                "balloon_bbox": [29, 96, 642, 709],
                            }
                        ],
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layers = project["paginas"][0]["text_layers"]
        self.assertGreaterEqual(audit["restored_missing_candidate_layers"], 1)
        self.assertIn("EI, VAMOS!", [layer.get("translated") for layer in layers])
        self.assertIn("ESTOU MORRENDO DE FOME", [layer.get("translated") for layer in layers])
        restored = [layer for layer in layers if layer.get("_restored_from_render_plan_candidate")]
        self.assertEqual(len(restored), 1)
        self.assertNotEqual(restored[0].get("id"), "ocr_001")

    def test_normalize_text_layer_removes_compact_inline_sfx_from_dialogue(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "ocr_001",
                "text": "DON'T HIT SFXKICK My MOM!",
                "original": "DON'T HIT SFXKICK My MOM!",
                "translated": "Não aperte sfxkick minha mãe!",
                "bbox": [10, 20, 180, 80],
            },
            page_number=1,
            layer_index=0,
        )

        self.assertEqual(layer["text"], "DON'T HIT My MOM!")
        self.assertEqual(layer["original"], "DON'T HIT My MOM!")
        self.assertEqual(layer["translated"], "Não aperte minha mãe!")
        self.assertEqual(layer["_inline_sfx_removed"], "KICK")

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

    def test_prepare_inpaint_base_restores_hidden_art_fragment_from_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            original_path = tmpdir / "original.png"
            inpainted_path = tmpdir / "inpainted.png"
            temp_output_path = tmpdir / "render-base.png"

            original = np.zeros((12, 12, 3), dtype=np.uint8)
            original[:, :] = (20, 60, 100)
            original[2:9, 2:9] = (210, 220, 230)
            inpainted = original.copy()
            inpainted[2:9, 2:9] = (0, 0, 0)
            cv2.imwrite(str(original_path), original)
            cv2.imwrite(str(inpainted_path), inpainted)

            with patch.dict("os.environ", {}, clear=True):
                result = main._prepare_inpaint_base_for_render(
                    original_path=original_path,
                    inpainted_path=inpainted_path,
                    texts=[
                        {
                            "text": "WU",
                            "translated": "WU",
                            "visible": False,
                            "route_action": "review_required",
                            "route_reason": "ocr_art_fragment_suspected",
                            "qa_flags": ["ocr_art_fragment_suspected"],
                            "bubble_mask_bbox": [2, 2, 9, 9],
                        }
                    ],
                    temp_output_path=temp_output_path,
                    update_inpaint=False,
                )

            self.assertEqual(result, temp_output_path)
            restored = cv2.imread(str(result))
            self.assertIsNotNone(restored)
            np.testing.assert_array_equal(restored[3, 3], original[3, 3])
            np.testing.assert_array_equal(cv2.imread(str(inpainted_path))[3, 3], inpainted[3, 3])

    def test_prepare_inpaint_base_cleans_false_dark_white_balloon_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            original_path = tmpdir / "original.png"
            inpainted_path = tmpdir / "inpainted.png"
            temp_output_path = tmpdir / "render-base.png"

            original = np.full((32, 48, 3), 255, dtype=np.uint8)
            inpainted = original.copy()
            inpainted[10:22, 14:34] = (0, 0, 0)
            cv2.imwrite(str(original_path), original)
            cv2.imwrite(str(inpainted_path), inpainted)

            result = main._prepare_inpaint_base_for_render(
                original_path=original_path,
                inpainted_path=inpainted_path,
                texts=[
                    {
                        "translated": "O QUE VOCE QUER DIZER?",
                        "layout_profile": "white_balloon",
                        "safe_text_box": [14, 10, 34, 22],
                        "qa_flags": [
                            "false_light_bubble_dark_fill_blocked",
                            "false_light_dark_bubble_promoted_to_white",
                            "false_dark_white_style_neutralized",
                        ],
                    }
                ],
                temp_output_path=temp_output_path,
                update_inpaint=False,
            )

            self.assertEqual(result, temp_output_path)
            cleaned = cv2.imread(str(result))
            self.assertIsNotNone(cleaned)
            self.assertGreater(int(cleaned[15, 20].mean()), 240)
            self.assertLess(int(cv2.imread(str(inpainted_path))[15, 20].mean()), 10)

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

    def test_render_preview_complete_emits_renderer_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = root / "originals"
            originals.mkdir()
            image_path = originals / "001.png"
            from PIL import Image

            Image.new("RGB", (32, 32), (255, 255, 255)).save(image_path)
            project_path = root / "project.json"
            page = {
                "numero": 1,
                "arquivo_original": "originals/001.png",
                "text_layers": [],
                "image_layers": {"base": {"path": "originals/001.png"}},
            }
            project_path.write_text(
                json.dumps({"paginas": [page], "_work_dir": str(root)}, ensure_ascii=False),
                encoding="utf-8",
            )
            override_path = root / "override.json"
            override_path.write_text(json.dumps({"page": page}, ensure_ascii=False), encoding="utf-8")
            output_path = root / "preview" / "001-preview.png"
            emitted = []

            def fake_typeset_single_page(args):
                _bg, _texts, out_dir = args
                out = Path(out_dir) / "001.png"
                Image.new("RGB", (32, 32), (255, 255, 255)).save(out)

            with patch.object(main, "emit", side_effect=lambda msg_type, **kwargs: emitted.append({"type": msg_type, **kwargs})):
                with patch("typesetter.renderer._typeset_single_page", side_effect=fake_typeset_single_page):
                    with patch.dict("os.environ", {"TRADUZAI_RENDERER_BACKEND": "koharu_rust"}):
                        main._run_render_preview_page(project_path, 0, override_path, output_path)

            complete = [event for event in emitted if event["type"] == "complete"][-1]
            self.assertEqual(complete["output_path"], str(output_path))
            self.assertEqual(complete["renderer_backend"], "koharu_rust")

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

    def test_refresh_debug_final_band_crops_uses_translated_image_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            image = np.zeros((10, 12, 3), dtype=np.uint8)
            image[2:7, 3:9] = [10, 80, 200]
            cv2.imwrite(str(translated_dir / "001.png"), image)
            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            debug_root.mkdir(parents=True)
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_001_band_000",
                        "translated_output_page": "001.png",
                        "crop_bbox_in_translated_page": [3, 2, 9, 7],
                        "final_crop_path": "10_copyback_reassemble/final_bands/page_001_band_000.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._refresh_debug_final_band_crops_from_translated(recorder, root)

            crop = cv2.imread(
                str(root / "debug" / "e2e" / "10_copyback_reassemble" / "final_bands" / "page_001_band_000.png"),
                cv2.IMREAD_COLOR,
            )
            self.assertEqual(audit["refreshed_count"], 1)
            self.assertIsNotNone(crop)
            self.assertEqual(crop.shape[:2], (5, 6))
            self.assertTrue(np.array_equal(crop, image[2:7, 3:9]))
            self.assertTrue((debug_root / "final_band_crops_refresh.json").exists())

    def test_post_rerender_visual_contract_restores_translated_from_final_bands_after_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            image = np.zeros((10, 12, 3), dtype=np.uint8)
            image[:, :] = [20, 20, 20]
            image[1:6, 2:8] = [32, 96, 220]
            cv2.imwrite(str(translated_dir / "001.png"), image)
            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            stale = np.zeros((5, 6, 3), dtype=np.uint8)
            stale[:, :] = [180, 20, 20]
            cv2.imwrite(str(final_bands / "page_001_band_000.png"), stale)
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_001_band_000",
                        "translated_output_page": "001.png",
                        "crop_bbox_in_translated_page": [2, 1, 8, 6],
                        "final_crop_path": "10_copyback_reassemble/final_bands/page_001_band_000.png",
                        "trace_ids": ["ocr_001@page_001_band_000"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_001_band_000",
                                "band_id": "page_001_band_000",
                                "translated": "Ola",
                                "text_pixel_bbox": [3, 2, 7, 5],
                                "render_bbox": [3, 2, 7, 5],
                                "safe_text_box": [2, 1, 8, 6],
                                "balloon_bbox": [2, 1, 8, 6],
                                "qa_flags": [],
                            }
                        ],
                    }
                ]
            }
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._run_post_rerender_final_visual_contract(
                recorder,
                project,
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            final_crop = cv2.imread(str(final_bands / "page_001_band_000.png"), cv2.IMREAD_COLOR)
            translated = cv2.imread(str(translated_dir / "001.png"), cv2.IMREAD_COLOR)
            self.assertEqual(audit["refresh"]["source"], "clean_final_bands_after_all_rerenders")
            self.assertTrue(audit["refresh"]["after_final_project_image_rerender"])
            self.assertTrue(audit["refresh"]["after_late_render_contract_repair"])
            self.assertEqual(audit["refresh"]["final_output_source"], "clean_final_bands_after_all_rerenders")
            self.assertEqual(audit["refresh"]["clean_band_source_used"], 1)
            self.assertLessEqual(float(np.mean(np.abs(final_crop.astype(np.int16) - stale.astype(np.int16)))), 8.0)
            self.assertLessEqual(float(np.mean(np.abs(translated[1:6, 2:8].astype(np.int16) - stale.astype(np.int16)))), 8.0)
            self.assertTrue((root / "debug" / "e2e" / "11_qa_export_gate" / "final_rerender_visual_qa.json").exists())
            self.assertTrue((root / "debug" / "e2e" / "11_qa_export_gate" / "final_rerender_visual_qa.jsonl").exists())

    def test_post_rerender_visual_contract_writes_translated_jpeg_with_low_recompression_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            page = np.zeros((18, 20, 3), dtype=np.uint8)
            page[:, :] = [20, 20, 20]
            cv2.imwrite(str(translated_dir / "001.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 100])

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            band = np.zeros((8, 10, 3), dtype=np.uint8)
            for y in range(8):
                for x in range(10):
                    band[y, x] = [20 + x * 7, 40 + y * 11, 180 - x * 3]
            cv2.imwrite(str(final_bands / "page_001_band_000.jpg"), band, [cv2.IMWRITE_JPEG_QUALITY, 100])
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_001_band_000",
                        "translated_output_page": "001.jpg",
                        "crop_bbox_in_translated_page": [4, 5, 14, 13],
                        "final_crop_path": "10_copyback_reassemble/final_bands/page_001_band_000.jpg",
                        "trace_ids": ["ocr_001@page_001_band_000"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            main._run_post_rerender_final_visual_contract(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": []}]},
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            final_crop = cv2.imread(str(final_bands / "page_001_band_000.jpg"), cv2.IMREAD_COLOR)
            translated = cv2.imread(str(translated_dir / "001.jpg"), cv2.IMREAD_COLOR)
            diff = np.abs(final_crop.astype(np.int16) - translated[5:13, 4:14].astype(np.int16))
            self.assertLessEqual(int(np.max(diff)), 12)

    def test_post_rerender_visual_contract_restores_clean_sources_after_stale_rerender(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            translated = np.zeros((16, 18, 3), dtype=np.uint8)
            translated[:, :] = [9, 9, 9]
            translated[4:10, 5:13] = [180, 10, 10]
            cv2.imwrite(str(translated_dir / "001.png"), translated)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            band_id = "page_002_band_023"
            clean = np.zeros((6, 8, 3), dtype=np.uint8)
            clean[:, :] = [11, 90, 210]
            post_dir = debug_root / band_id
            post_dir.mkdir(parents=True)
            cv2.imwrite(str(post_dir / "post_copyback.png"), clean)
            rendered_dir = root / "debug" / "e2e" / "09_typeset" / "rendered_bands"
            rendered_dir.mkdir(parents=True)
            cv2.imwrite(str(rendered_dir / f"{band_id}.png"), clean)
            stale_final = np.zeros((6, 8, 3), dtype=np.uint8)
            stale_final[:, :] = [230, 230, 230]
            cv2.imwrite(str(final_bands / f"{band_id}.png"), stale_final)
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": band_id,
                        "translated_output_page": "001.png",
                        "crop_bbox_in_translated_page": [5, 4, 13, 10],
                        "final_crop_path": f"10_copyback_reassemble/final_bands/{band_id}.png",
                        "rendered_band_path": f"09_typeset/rendered_bands/{band_id}.png",
                        "trace_ids": [f"ocr_001@{band_id}"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_traduzido": "translated/001.png",
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": f"ocr_001@{band_id}",
                                "band_id": band_id,
                                "translated": "Texto limpo",
                                "text_pixel_bbox": [6, 5, 12, 9],
                                "render_bbox": [6, 5, 12, 9],
                                "safe_text_box": [5, 4, 13, 10],
                                "balloon_bbox": [5, 4, 13, 10],
                                "qa_flags": [],
                            }
                        ],
                    }
                ]
            }
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._run_post_rerender_final_visual_contract(
                recorder,
                project,
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            final_crop = cv2.imread(str(final_bands / f"{band_id}.png"), cv2.IMREAD_COLOR)
            translated_after = cv2.imread(str(translated_dir / "001.png"), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(final_crop, clean))
            self.assertTrue(np.array_equal(translated_after[4:10, 5:13], clean))
            self.assertEqual(audit["refresh"]["source"], "clean_final_bands_after_all_rerenders")
            self.assertTrue(audit["refresh"]["final_guard_ran_after_final_project_image_rerender"])
            self.assertEqual(audit["refresh"]["clean_band_source_used"], 1)
            self.assertEqual(audit["refresh"]["clean_band_final_mismatch_count"], 1)
            self.assertEqual(audit["refresh"]["final_output_source"], "clean_final_bands_after_all_rerenders")

    def test_final_band_refresh_records_excluded_non_story_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            debug_root.mkdir(parents=True)
            (debug_root / "non_story_exclusions.json").write_text(
                json.dumps(
                    {
                        "excluded_non_story_bands": ["page_005_band_102"],
                        "excluded_count": 1,
                        "exclusions": [
                            {
                                "band_id": "page_005_band_102",
                                "export_policy": "exclude_from_translated_output",
                                "exclusion_reason": "scanlation_discord_promo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (debug_root / "final_band_crops.jsonl").write_text("", encoding="utf-8")

            recorder = DebugRecorder(root, enabled=True, run_id="run-test")
            audit = main._restore_clean_final_bands_after_rerender(recorder, root)

            self.assertEqual(audit["excluded_non_story_bands"], ["page_005_band_102"])
            self.assertEqual(
                audit["excluded_non_story_reasons"],
                {"page_005_band_102": "scanlation_discord_promo"},
            )

    def test_post_rerender_visual_contract_prioritizes_text_bands_over_empty_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            translated = np.zeros((14, 12, 3), dtype=np.uint8)
            translated[:, :] = [4, 4, 4]
            cv2.imwrite(str(translated_dir / "001.png"), translated)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            rendered_dir = root / "debug" / "e2e" / "09_typeset" / "rendered_bands"
            rendered_dir.mkdir(parents=True)

            empty_band = np.zeros((6, 12, 3), dtype=np.uint8)
            empty_band[:, :] = [180, 20, 20]
            text_band = np.zeros((6, 12, 3), dtype=np.uint8)
            text_band[:, :] = [20, 180, 80]
            cv2.imwrite(str(rendered_dir / "page_003_band_045.png"), empty_band)
            text_post_dir = debug_root / "page_003_band_046"
            text_post_dir.mkdir(parents=True)
            cv2.imwrite(str(text_post_dir / "post_copyback.png"), text_band)
            cv2.imwrite(str(final_bands / "page_003_band_045.png"), empty_band)
            cv2.imwrite(str(final_bands / "page_003_band_046.png"), text_band)
            rows = [
                {
                    "band_id": "page_003_band_045",
                    "translated_output_page": "001.png",
                    "band_y_top": 100,
                    "band_y_bottom": 106,
                    "crop_bbox_in_translated_page": [0, 4, 12, 10],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_003_band_045.png",
                    "rendered_band_path": "09_typeset/rendered_bands/page_003_band_045.png",
                    "trace_ids": [],
                },
                {
                    "band_id": "page_003_band_046",
                    "translated_output_page": "001.png",
                    "band_y_top": 102,
                    "band_y_bottom": 108,
                    "crop_bbox_in_translated_page": [0, 6, 12, 12],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_003_band_046.png",
                    "rendered_band_path": "09_typeset/rendered_bands/page_003_band_046.png",
                    "trace_ids": ["ocr_001@page_003_band_046"],
                },
            ]
            (debug_root / "final_band_crops.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            main._run_post_rerender_final_visual_contract(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": []}]},
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            translated_after = cv2.imread(str(translated_dir / "001.png"), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(translated_after[6:10, 0:12], text_band[0:4, 0:12]))

    def test_post_rerender_visual_contract_keeps_upper_post_copyback_over_lower_trace_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            page = np.zeros((14, 12, 3), dtype=np.uint8)
            page[:, :] = [4, 4, 4]
            cv2.imwrite(str(translated_dir / "001.png"), page)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            upper = np.zeros((6, 12, 3), dtype=np.uint8)
            upper[:, :] = [20, 180, 80]
            lower = np.zeros((6, 12, 3), dtype=np.uint8)
            lower[:, :] = [180, 20, 20]
            for band_id, image in [("page_004_band_055", upper), ("page_004_band_056", lower)]:
                post_dir = debug_root / band_id
                post_dir.mkdir(parents=True)
                cv2.imwrite(str(post_dir / "post_copyback.png"), image)
                cv2.imwrite(str(final_bands / f"{band_id}.png"), image)
            rows = [
                {
                    "band_id": "page_004_band_055",
                    "translated_output_page": "001.png",
                    "band_y_top": 100,
                    "band_y_bottom": 106,
                    "crop_bbox_in_translated_page": [0, 4, 12, 10],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_004_band_055.png",
                    "trace_ids": [],
                },
                {
                    "band_id": "page_004_band_056",
                    "translated_output_page": "001.png",
                    "band_y_top": 102,
                    "band_y_bottom": 108,
                    "crop_bbox_in_translated_page": [0, 6, 12, 12],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_004_band_056.png",
                    "trace_ids": ["ocr_002@page_004_band_056"],
                },
            ]
            (debug_root / "final_band_crops.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            main._run_post_rerender_final_visual_contract(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": []}]},
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            translated_after = cv2.imread(str(translated_dir / "001.png"), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(translated_after[6:10, 0:12], upper[2:6, 0:12]))

    def test_translated_page_band_consistency_compares_only_visible_overlap_owner_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            page = np.zeros((14, 12, 3), dtype=np.uint8)
            page[:, :] = [4, 4, 4]
            cv2.imwrite(str(translated_dir / "001.png"), page)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)

            upper = np.zeros((6, 12, 3), dtype=np.uint8)
            upper[:, :] = [180, 20, 20]
            lower = np.zeros((6, 12, 3), dtype=np.uint8)
            lower[:, :] = [20, 180, 80]
            cv2.imwrite(str(final_bands / "page_003_band_045.png"), upper)
            lower_post = debug_root / "page_003_band_046"
            lower_post.mkdir(parents=True)
            cv2.imwrite(str(lower_post / "post_copyback.png"), lower)
            cv2.imwrite(str(final_bands / "page_003_band_046.png"), lower)

            rows = [
                {
                    "band_id": "page_003_band_045",
                    "translated_output_page": "001.png",
                    "band_y_top": 100,
                    "band_y_bottom": 106,
                    "crop_bbox_in_translated_page": [0, 4, 12, 10],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_003_band_045.png",
                    "trace_ids": [],
                },
                {
                    "band_id": "page_003_band_046",
                    "translated_output_page": "001.png",
                    "band_y_top": 102,
                    "band_y_bottom": 108,
                    "crop_bbox_in_translated_page": [0, 6, 12, 12],
                    "final_crop_path": "10_copyback_reassemble/final_bands/page_003_band_046.png",
                    "trace_ids": ["ocr_001@page_003_band_046"],
                },
            ]
            (debug_root / "final_band_crops.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._run_post_rerender_final_visual_contract(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": []}]},
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )["translated_page_band_consistency"]

            self.assertTrue(audit["passed"])
            self.assertEqual(audit["rows_failed"], 0)
            rows_by_band = {row["band_id"]: row for row in audit["rows"]}
            self.assertEqual(rows_by_band["page_003_band_045"]["visible_pixels"], 24)
            self.assertEqual(rows_by_band["page_003_band_045"]["ignored_overlap_pixels"], 48)
            self.assertEqual(rows_by_band["page_003_band_045"]["overlap_owner_band_id"], "page_003_band_046")
            self.assertEqual(rows_by_band["page_003_band_045"]["changed_gt8"], 0)
            self.assertEqual(rows_by_band["page_003_band_046"]["visible_pixels"], 72)
            self.assertTrue((debug_root / "translated_page_band_consistency_audit.json").exists())

    def test_post_rerender_visual_contract_uses_visible_slice_for_page_start_clipped_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            page = np.zeros((8, 10, 3), dtype=np.uint8)
            page[:, :] = [1, 1, 1]
            cv2.imwrite(str(translated_dir / "001.png"), page)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            band_id = "page_002_band_018"
            full_band = np.zeros((6, 10, 3), dtype=np.uint8)
            full_band[0:2, :] = [200, 20, 20]
            full_band[2:6, :] = [20, 180, 80]
            post_dir = debug_root / band_id
            post_dir.mkdir(parents=True)
            cv2.imwrite(str(post_dir / "post_copyback.png"), full_band)
            cv2.imwrite(str(final_bands / f"{band_id}.png"), full_band)
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": band_id,
                        "translated_output_page": "001.png",
                        "output_page_y_top": 102,
                        "band_y_top": 100,
                        "band_y_bottom": 106,
                        "crop_bbox_in_translated_page": [0, 0, 10, 4],
                        "final_crop_path": f"10_copyback_reassemble/final_bands/{band_id}.png",
                        "trace_ids": ["ocr_001@page_002_band_018"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            main._run_post_rerender_final_visual_contract(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": []}]},
                root,
                after_final_project_image_rerender=True,
                after_late_render_contract_repair=True,
            )

            translated_after = cv2.imread(str(translated_dir / "001.png"), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(translated_after[0:4, 0:10], full_band[2:6, 0:10]))

    def test_post_rerender_visual_contract_keeps_final_bands_when_no_rerender_happened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            translated = np.zeros((10, 12, 3), dtype=np.uint8)
            translated[:, :] = [20, 20, 20]
            translated[1:6, 2:8] = [32, 96, 220]
            cv2.imwrite(str(translated_dir / "001.jpg"), translated, [cv2.IMWRITE_JPEG_QUALITY, 100])
            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            original_final = np.zeros((5, 6, 3), dtype=np.uint8)
            original_final[:, :] = [180, 20, 20]
            cv2.imwrite(str(final_bands / "page_001_band_000.jpg"), original_final, [cv2.IMWRITE_JPEG_QUALITY, 100])
            (debug_root / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_001_band_000",
                        "translated_output_page": "001.jpg",
                        "crop_bbox_in_translated_page": [2, 1, 8, 6],
                        "final_crop_path": "10_copyback_reassemble/final_bands/page_001_band_000.jpg",
                        "trace_ids": ["ocr_001@page_001_band_000"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_001_band_000",
                                "band_id": "page_001_band_000",
                                "translated": "Ola",
                                "text_pixel_bbox": [3, 2, 7, 5],
                                "render_bbox": [3, 2, 7, 5],
                                "safe_text_box": [2, 1, 8, 6],
                                "balloon_bbox": [2, 1, 8, 6],
                                "qa_flags": [],
                            }
                        ],
                    }
                ]
            }
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._run_post_rerender_final_visual_contract(
                recorder,
                project,
                root,
                after_final_project_image_rerender=False,
                after_late_render_contract_repair=True,
            )

            final_crop = cv2.imread(str(final_bands / "page_001_band_000.jpg"), cv2.IMREAD_COLOR)
            self.assertEqual(audit["refresh"]["refreshed_count"], 0)
            self.assertTrue(audit["refresh"]["skipped_no_final_rerender"])
            self.assertTrue(audit["refresh"]["after_late_render_contract_repair"])
            self.assertLessEqual(float(np.mean(np.abs(final_crop.astype(np.int16) - original_final.astype(np.int16)))), 3.0)

    def test_final_rerender_visual_qa_flags_dark_bubble_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            image = np.full((220, 120, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(translated_dir / "001.png"), image)
            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            crop_rows = []
            for index, band_id in enumerate(
                [
                    "page_001_band_oval",
                    "page_001_band_short",
                    "page_001_band_lobe_a",
                    "page_001_band_lobe_b",
                    "page_001_band_panel",
                ]
            ):
                y1 = index * 40
                y2 = y1 + 36
                crop = image[y1:y2, 0:90].copy()
                final_path = final_bands / f"{band_id}.png"
                cv2.imwrite(str(final_path), crop)
                crop_rows.append(
                    {
                        "band_id": band_id,
                        "translated_output_page": "001.png",
                        "crop_bbox_in_translated_page": [0, y1, 90, y2],
                        "final_crop_path": f"10_copyback_reassemble/final_bands/{band_id}.png",
                        "trace_ids": [f"ocr_{index + 1:03}@{band_id}"],
                    }
                )
            (debug_root / "final_band_crops.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in crop_rows),
                encoding="utf-8",
            )
            layers = [
                {
                    "id": "ocr_001",
                    "trace_id": "ocr_001@page_001_band_oval",
                    "band_id": "page_001_band_oval",
                    "translated": "Normal",
                    "text_pixel_bbox": [20, 8, 70, 28],
                    "render_bbox": [22, 10, 68, 27],
                    "safe_text_box": [10, 4, 80, 32],
                    "balloon_bbox": [5, 2, 85, 34],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["dark_oval_safe_height_expanded"],
                },
                {
                    "id": "ocr_002",
                    "trace_id": "ocr_002@page_001_band_short",
                    "band_id": "page_001_band_short",
                    "translated": "1.000 pontos",
                    "text_pixel_bbox": [15, 48, 75, 72],
                    "render_bbox": [42, 57, 48, 60],
                    "safe_text_box": [10, 44, 80, 76],
                    "balloon_bbox": [5, 42, 85, 78],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "qa_flags": ["dark_oval_safe_height_expanded"],
                },
                {
                    "id": "ocr_003",
                    "trace_id": "ocr_003@page_001_band_lobe_a",
                    "band_id": "page_001_band_lobe_a",
                    "translated": "Lobe A",
                    "text_pixel_bbox": [18, 88, 70, 112],
                    "render_bbox": [20, 90, 68, 111],
                    "safe_text_box": [10, 84, 80, 116],
                    "balloon_bbox": [5, 82, 85, 118],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "dark_connected_group_id": "dark_pair_1",
                    "qa_flags": ["dark_panel_style_grouped"],
                },
                {
                    "id": "ocr_004",
                    "trace_id": "ocr_004@page_001_band_lobe_b",
                    "band_id": "page_001_band_lobe_b",
                    "translated": "Lobe B",
                    "text_pixel_bbox": [18, 128, 70, 152],
                    "render_bbox": [4, 124, 16, 130],
                    "safe_text_box": [10, 124, 80, 156],
                    "balloon_bbox": [5, 122, 85, 158],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "layout_profile": "dark_bubble",
                    "dark_connected_group_id": "dark_pair_1",
                    "qa_flags": ["dark_panel_style_grouped"],
                },
                {
                    "id": "ocr_005",
                    "trace_id": "ocr_005@page_001_band_panel",
                    "band_id": "page_001_band_panel",
                    "translated": "Panel",
                    "text_pixel_bbox": [20, 168, 70, 190],
                    "render_bbox": [82, 162, 116, 198],
                    "safe_text_box": [10, 164, 80, 196],
                    "balloon_bbox": [5, 162, 85, 198],
                    "bubble_mask_source": "image_dark_panel_mask",
                    "layout_profile": "dark_panel",
                    "qa_flags": [],
                },
            ]
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._qa_translated_final_crops_against_layers(recorder, {"paginas": [{"numero": 1, "text_layers": layers}]}, root)

            rows = {row["band_id"]: row for row in audit["rows"]}
            self.assertEqual(rows["page_001_band_oval"]["status"], "pass")
            self.assertIn("dark_text_tiny", rows["page_001_band_short"]["flags"])
            self.assertIn("dark_connected_lobe_issue", rows["page_001_band_lobe_b"]["flags"])
            self.assertIn("dark_text_outside_balloon", rows["page_001_band_panel"]["flags"])
            self.assertIn("dark_text_tiny_ratio", rows["page_001_band_short"]["metrics"])
            self.assertIn("dark_text_center_drift", rows["page_001_band_lobe_b"]["metrics"])
            self.assertIn("dark_text_outside_balloon_ratio", rows["page_001_band_panel"]["metrics"])
            self.assertTrue((root / "debug" / "e2e" / "11_qa_export_gate" / "final_rerender_visual_qa.json").exists())

    def test_final_rerender_visual_qa_flags_dark_oval_underfilled_medium_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from debug_tools import DebugRecorder

            root = Path(tmp)
            translated_dir = root / "translated"
            translated_dir.mkdir(parents=True)
            image = np.full((80, 140, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(translated_dir / "001.png"), image)

            debug_root = root / "debug" / "e2e" / "10_copyback_reassemble"
            final_bands = debug_root / "final_bands"
            final_bands.mkdir(parents=True)
            band_id = "page_001_band_dark_underfilled"
            final_path = final_bands / f"{band_id}.png"
            cv2.imwrite(str(final_path), image[0:72, 0:128].copy())
            crop_row = {
                "band_id": band_id,
                "translated_output_page": "001.png",
                "crop_bbox_in_translated_page": [0, 0, 128, 72],
                "final_crop_path": f"10_copyback_reassemble/final_bands/{band_id}.png",
                "trace_ids": [f"ocr_001@{band_id}"],
            }
            (debug_root / "final_band_crops.jsonl").write_text(json.dumps(crop_row) + "\n", encoding="utf-8")

            layer = {
                "id": "ocr_001",
                "trace_id": f"ocr_001@{band_id}",
                "band_id": band_id,
                "translated": "AINDA CONSIGO SENTIR A PRESENCA DELE",
                "text_pixel_bbox": [24, 12, 96, 52],
                "render_bbox": [36, 15, 84, 39],
                "safe_text_box": [10, 8, 110, 58],
                "balloon_bbox": [4, 4, 116, 64],
                "bubble_mask_source": "image_dark_bubble_mask",
                "layout_profile": "dark_bubble",
                "qa_flags": ["dark_oval_safe_height_expanded"],
            }
            recorder = DebugRecorder(root, enabled=True, run_id="run-test")

            audit = main._qa_translated_final_crops_against_layers(
                recorder,
                {"paginas": [{"numero": 1, "text_layers": [layer]}]},
                root,
            )

            row = audit["rows"][0]
            self.assertGreater(row["metrics"]["layers"][0]["tiny_text_ratio"], 0.35)
            self.assertNotIn("tiny_text", row["flags"])
            self.assertNotIn("dark_text_tiny", row["flags"])
            self.assertEqual(row["status"], "fail")
            self.assertIn("dark_text_underfilled", row["flags"])
            self.assertIn("dark_text_underfilled_height_ratio", row["metrics"])
            self.assertIn("dark_text_underfilled_area_ratio", row["metrics"])

    def test_persist_real_bubble_mask_layer_rejects_bbox_fallback_source(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            page = {"numero": 1, "text_layers": [{"id": "t1", "bubble_id": "b1", "bbox": [10, 10, 30, 30]}]}
            ocr_page = {
                "width": 40,
                "height": 40,
                "_bubble_regions": [
                    {
                        "bubble_id": "b1",
                        "bubble_mask_source": "bbox_fallback",
                        "bubble_mask_bbox": [5, 5, 35, 35],
                        "bubble_mask": np.ones((30, 30), dtype=np.uint8) * 255,
                    }
                ],
            }

            persisted = main._persist_real_bubble_mask_layer_for_page(
                page,
                ocr_page,
                Path(tmp),
                page_number=1,
                image_size=(40, 40),
            )

            self.assertFalse(persisted)
            self.assertNotIn("image_layers", page)
            self.assertFalse((Path(tmp) / "layers" / "bubble-mask" / "001.png").exists())
            self.assertNotIn("bubble_mask_value", page["text_layers"][0])

    def test_persist_real_bubble_mask_layer_rejects_image_derived_source(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            page = {"numero": 1, "text_layers": [{"id": "t1", "bubble_id": "b1", "bbox": [10, 10, 30, 30]}]}
            ocr_page = {
                "width": 40,
                "height": 40,
                "_bubble_regions": [
                    {
                        "bubble_id": "b1",
                        "bubble_mask_source": "image_white_bubble_mask",
                        "bubble_mask_bbox": [5, 5, 35, 35],
                        "bubble_mask": np.ones((30, 30), dtype=np.uint8) * 255,
                    }
                ],
            }

            persisted = main._persist_real_bubble_mask_layer_for_page(
                page,
                ocr_page,
                Path(tmp),
                page_number=1,
                image_size=(40, 40),
            )

            self.assertFalse(persisted)
            self.assertNotIn("image_layers", page)
            self.assertFalse((Path(tmp) / "layers" / "bubble-mask" / "001.png").exists())
            self.assertNotIn("bubble_mask_value", page["text_layers"][0])

    def test_normalized_text_layers_preserve_bubble_mask_source(self) -> None:
        layer = main._normalize_text_layer_for_renderer(
            {
                "id": "t1",
                "bbox": [10, 10, 40, 30],
                "translated": "OLA",
                "bubble_id": "b1",
                "bubble_mask_bbox": [0, 0, 80, 60],
                "bubble_mask_source": "bbox_fallback",
                "bubble_mask_error": "missing_real_bubble_mask",
            },
            page_number=1,
            layer_index=0,
        )

        self.assertEqual(layer["bubble_mask_source"], "bbox_fallback")
        self.assertEqual(layer["bubble_mask_error"], "missing_real_bubble_mask")

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

    def test_run_translate_page_enriches_sfx_and_excludes_it_from_dialogue_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project.json"
            translated_dir = Path(tmp) / "translated"
            translated_dir.mkdir()
            (translated_dir / "001.jpg").write_bytes(b"fake")

            project = {
                "idioma_origem": "ko",
                "idioma_destino": "pt-BR",
                "obra": "Teste",
                "contexto": {"sinopse": "abc"},
                "paginas": [
                    {
                        "numero": 1,
                        "arquivo_traduzido": "translated/001.jpg",
                        "text_layers": [
                            {
                                "id": "sfx-1",
                                "bbox": [1, 2, 30, 40],
                                "original": "\ucff5",
                                "text": "\ucff5",
                                "tipo": "sfx",
                                "content_class": "sfx",
                                "script": "hangul",
                                "route_action": "translate_sfx_inpaint_render",
                                "translate_policy": "adapt_sfx",
                                "render_policy": "sfx_style",
                            },
                            {
                                "id": "dialogue-1",
                                "bbox": [40, 50, 90, 110],
                                "original": "HELLO",
                                "traduzido": "",
                                "tipo": "fala",
                            },
                        ],
                        "textos": [],
                    }
                ],
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            def fake_translate_pages(**kwargs):
                texts = kwargs["ocr_results"][0]["texts"]
                self.assertEqual([text["id"] for text in texts], ["dialogue-1"])
                self.assertEqual(texts[0]["text"], "HELLO")
                return [
                    {
                        "texts": [
                            {
                                **texts[0],
                                "translated": "OLÃ",
                                "traduzido": "OLÃ",
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
            sfx_layer, dialogue_layer = saved["paginas"][0]["text_layers"]

            self.assertEqual(sfx_layer["content_class"], "sfx")
            self.assertEqual(sfx_layer["script"], "hangul")
            self.assertEqual(sfx_layer["route_action"], "translate_sfx_inpaint_render")
            self.assertEqual(sfx_layer["translate_policy"], "adapt_sfx")
            self.assertEqual(sfx_layer["render_policy"], "sfx_style")
            self.assertEqual(sfx_layer["translated"], "TUM")
            self.assertEqual(sfx_layer["traduzido"], "TUM")
            self.assertEqual(sfx_layer["sfx"]["source_text"], "\ucff5")
            self.assertEqual(sfx_layer["sfx"]["adapted_text"], "TUM")
            self.assertEqual(sfx_layer["sfx"]["translation_mode"], "onomatopoeia_adaptation")
            self.assertFalse(sfx_layer["sfx"]["inpaint_allowed"])
            self.assertEqual(dialogue_layer["translated"], "OLÃ")
            self.assertEqual(saved["paginas"][0]["textos"][0]["id"], "sfx-1")

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
                schema_version=5,
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
                    "background_rgb": [184, 196, 224],
                    "ui_layout_evidence": {
                        "source": "uied_cv",
                        "role": "text_inside_component",
                        "component_bbox": [10, 20, 190, 130],
                        "confidence": 0.82,
                    },
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
        self.assertEqual(legacy["background_rgb"], [184, 196, 224])
        self.assertEqual(legacy["ui_layout_evidence"]["source"], "uied_cv")
        self.assertEqual(legacy["balloon_type"], "")

    def test_build_text_layer_prefers_ui_form_profile_when_uied_evidence_exists(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "ocr_ui",
                "text": "Open status window",
                "bbox": [40, 50, 180, 70],
                "confidence": 0.91,
                "layout_profile": "connected_balloon",
                "block_profile": "white_balloon",
                "ui_layout_evidence": {
                    "source": "uied_cv",
                    "role": "text_inside_component",
                    "component_bbox": [20, 40, 220, 82],
                    "confidence": 0.84,
                },
            },
            translated="Abrir janela de status",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["layout_profile"], "ui_form")
        self.assertEqual(layer["block_profile"], "ui_form")
        self.assertEqual(layer["ui_layout_evidence"]["source"], "uied_cv")

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

    def test_build_text_layer_applies_high_confidence_source_style_evidence(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.82,
            "stroke_color": "#000000",
            "stroke_width_px": 3,
            "stroke_confidence": 0.78,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.74,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "BOOM",
                "bbox": [0, 0, 100, 100],
                "tipo": "sfx",
                "confidence": 0.92,
                "background_rgb": [30, 30, 30],
                "style_evidence": evidence,
            },
            translated="BUM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_confidence"], 0.82)
        self.assertEqual(layer["style_source"], "pixel_analysis")
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertIs(layer["style"], layer["estilo"])
        self.assertEqual(layer["estilo"]["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["style_source"], "pixel_analysis")
        self.assertEqual(layer["estilo"]["fonte"], "KOMIKAX_.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#FFFFFF")
        self.assertEqual(layer["estilo"]["contorno"], "#000000")
        self.assertEqual(layer["estilo"]["contorno_px"], 3)

    def test_build_text_layer_extracts_source_style_evidence_from_image_crop(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        source_image_rgb[10:40, 20:70] = [255, 255, 255]
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.82,
            "stroke_color": "#000000",
            "stroke_width_px": 2,
            "stroke_confidence": 0.78,
        }
        crop_shapes = []

        def fake_extract(crop, **kwargs):
            del kwargs
            crop_shapes.append(tuple(crop.shape))
            self.assertTrue(np.all(crop == 255))
            return evidence

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fake_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "tl_001_001",
                    "text": "BOOM",
                    "bbox": [18, 8, 72, 42],
                    "text_pixel_bbox": [20, 10, 70, 40],
                    "tipo": "sfx",
                    "confidence": 0.92,
                    "background_rgb": [30, 30, 30],
                },
                translated="BUM",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertEqual(crop_shapes, [(30, 50, 3)])
        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_source"], "pixel_analysis")
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["cor"], "#FFFFFF")
        self.assertEqual(layer["estilo"]["contorno"], "#000000")
        self.assertEqual(layer["estilo"]["contorno_px"], 2)

    def test_build_text_layer_passes_font_detector_to_source_style_scan(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
        source_image_rgb[10:40, 20:70] = [255, 255, 255]
        detector = object()
        seen_detectors = []

        def fake_extract(crop, *, font_detector=None, font_context=None):
            del crop
            seen_detectors.append((font_detector, font_context))
            return {
                "source": "pixel_analysis",
                "text_color": "#FFFFFF",
                "text_color_confidence": 0.82,
                "stroke_color": "#000000",
                "stroke_width_px": 2,
                "stroke_confidence": 0.78,
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_confidence": 0.88,
            }

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fake_extract):
            layer = main.build_text_layer(
                page_number=6,
                layer_index=0,
                ocr_text={
                    "id": "tl_006_001",
                    "text": "Synching is complete.",
                    "bbox": [18, 8, 72, 42],
                    "text_pixel_bbox": [20, 10, 70, 40],
                    "tipo": "text",
                    "confidence": 0.96,
                    "background_rgb": [92, 154, 224],
                    "bubble_mask_source": "image_white_bubble_mask",
                },
                translated="A sincronização foi concluída.",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
                font_detector=detector,
            )

        self.assertEqual(seen_detectors, [(detector, "visual_card")])
        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")

    def test_build_text_layer_scans_review_required_renderable_text_for_source_glow_style(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((80, 160, 3), dtype=np.uint8)
        source_image_rgb[20:52, 30:132] = [8, 18, 24]
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.72,
            "glow": True,
            "glow_color": "#74D7FF",
            "glow_px": 4,
            "glow_confidence": 0.86,
        }

        with patch("typesetter.style_extractor.extract_text_style_evidence", return_value=evidence):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "tl_001_001",
                    "text": "The Devil Knight!",
                    "bbox": [30, 20, 132, 52],
                    "text_pixel_bbox": [30, 20, 132, 52],
                    "route_action": "review_required",
                    "render_policy": "review_required",
                    "confidence": 0.0,
                    "ocr_confidence": 0.0,
                    "background_rgb": [8, 18, 24],
                },
                translated="O cavaleiro do diabo!",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_confidence"], 0.86)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["cor"], "#FFFFFF")
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#74D7FF")
        self.assertEqual(layer["estilo"]["glow_px"], 4)

    def test_build_text_layer_applies_high_confidence_shadow_glow_style_evidence(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#F8E8FF",
            "text_color_confidence": 0.33,
            "shadow": True,
            "shadow_confidence": 0.72,
            "shadow_color": "#111111",
            "shadow_offset": [3, 4],
            "glow": True,
            "glow_confidence": 0.76,
            "glow_px": 4,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "FLASH",
                "bbox": [0, 0, 100, 100],
                "tipo": "sfx",
                "confidence": 0.92,
                "background_rgb": [30, 30, 30],
                "style_evidence": evidence,
            },
            translated="FLASH",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_confidence"], 0.76)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertTrue(layer["estilo"]["sombra"])
        self.assertEqual(layer["estilo"]["sombra_cor"], "#111111")
        self.assertEqual(layer["estilo"]["sombra_offset"], [3, 4])
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#F8E8FF")
        self.assertEqual(layer["estilo"]["glow_px"], 4)

    def test_build_text_layer_keeps_visual_sfx_style_evidence_but_does_not_apply_without_text(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.94,
            "stroke_color": "#001144",
            "stroke_width_px": 4,
            "stroke_confidence": 0.93,
            "gradient": True,
            "gradient_colors": ["#FFFFFF", "#78D7FF"],
            "gradient_confidence": 0.91,
            "glow": True,
            "glow_color": "#BCEEFF",
            "glow_confidence": 0.9,
            "glow_px": 5,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.88,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_001",
                "text": "",
                "bbox": [20, 20, 180, 120],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "route_action": "translate_sfx_inpaint_render",
                "route_reason": "visual_sfx_promoted_without_ocr",
                "translate_policy": "review",
                "render_policy": "sfx_style",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "visual_promotion": True,
                    "source_text": "",
                    "adapted_text": "",
                    "translation_mode": "visual_sfx_manual_text_required",
                    "inpaint_allowed": False,
                },
            },
            translated="",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.94)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#000000")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)
        self.assertEqual(layer["estilo"].get("cor_gradiente"), [])
        self.assertFalse(layer["estilo"]["glow"])
        self.assertEqual(layer["sfx"]["translation_mode"], "visual_sfx_manual_text_required")

    def test_build_text_layer_does_not_apply_style_to_visual_sfx_review_even_with_partial_ocr(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.96,
            "stroke_color": "#101010",
            "stroke_width_px": 3,
            "stroke_confidence": 0.91,
            "glow": True,
            "glow_color": "#FFFFFF",
            "glow_confidence": 0.9,
            "glow_px": 4,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.86,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_ocr_noise",
                "text": "마",
                "bbox": [20, 20, 180, 120],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "route_action": "review_required",
                "translate_policy": "review",
                "render_policy": "review_required",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "source_text": "",
                    "adapted_text": "",
                    "inpaint_allowed": False,
                },
            },
            translated="마",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.96)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertFalse(layer["estilo"]["glow"])

    def test_build_text_layer_neutralizes_style_when_sfx_enrichment_routes_to_review(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.96,
            "stroke_color": "#101010",
            "stroke_width_px": 3,
            "stroke_confidence": 0.91,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.86,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_late_review",
                "text": "마",
                "bbox": [20, 20, 180, 120],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "source_text": "",
                    "adapted_text": "",
                    "inpaint_allowed": False,
                },
            },
            translated="마",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["render_policy"], "review_required")
        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.96)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_does_not_apply_style_for_review_sfx_even_when_text_is_latin_phrase(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#161205",
            "text_color_confidence": 1.0,
            "gradient": True,
            "gradient_colors": ["#2F2407", "#0B0A05"],
            "gradient_confidence": 0.88,
            "font_name": "ComicNeue-Bold.ttf",
            "font_confidence": 1.0,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_latin_text",
                "text": "MAINTAINED HIS POSITION AS A HIGH-RANKER.",
                "bbox": [20, 20, 220, 140],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "source_text": "",
                    "adapted_text": "",
                    "inpaint_allowed": False,
                },
            },
            translated="MAINTAINED HIS POSITION AS A HIGH-RANKER.",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"].get("cor_gradiente"), [])

    def test_build_text_layer_does_not_use_translated_text_to_allow_review_sfx_style(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.95,
            "stroke_color": "#101010",
            "stroke_width_px": 4,
            "stroke_confidence": 0.9,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_empty_source_with_translated_leak",
                "text": "",
                "bbox": [20, 20, 220, 140],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "route_action": "review_required",
                "render_policy": "review_required",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_text_detector",
                    "source_text": "",
                    "adapted_text": "",
                    "inpaint_allowed": False,
                },
            },
            translated="CONSTELAÇÃO ATENA MOSTRA INTERESSE PELO RELÂMPAGO DO OLIMPO",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.95)
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_does_not_apply_source_style_to_scanlator_watermark_text(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#E4E4E4",
            "text_color_confidence": 0.86,
            "stroke_color": "#030303",
            "stroke_width_px": 2,
            "stroke_confidence": 0.82,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.5,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "scanlator_text",
                "text": "YOU CAN READ THE CHAPTER ON: EN-THUNDERSCANS.COM",
                "bbox": [20, 20, 360, 80],
                "content_class": "text",
                "tipo": "text",
                "route_action": "translate_inpaint_render",
                "style_evidence": evidence,
            },
            translated="YOU CAN READ THE CHAPTER ON: EN-THUNDERSCANS.COM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.86)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_allows_large_dark_text_with_light_outline_without_ocr(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#010100",
            "text_color_confidence": 0.99,
            "stroke_color": "#FFFFFE",
            "stroke_width_px": 2,
            "stroke_confidence": 0.99,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.5,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_large_outline_text",
                "text": "",
                "bbox": [60, 5600, 490, 5800],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "route_action": "review_required",
                "render_policy": "review_required",
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_text_detector",
                    "source_text": "",
                    "adapted_text": "",
                    "inpaint_allowed": False,
                },
            },
            translated="",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["fonte"], "KOMIKAX_.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#010100")
        self.assertEqual(layer["estilo"]["contorno"], "#FFFFFE")
        self.assertEqual(layer["estilo"]["contorno_px"], 2)

    def test_build_text_layer_does_not_extract_evidence_for_large_review_sfx_visual(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((240, 520, 3), dtype=np.uint8)

        def fake_extract(crop):
            raise AssertionError("style extractor should not run for review-only SFX visuals")

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fake_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "sfx_visual_large_outline_text_from_crop",
                    "text": "",
                    "bbox": [40, 30, 480, 190],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "route_action": "review_required",
                    "render_policy": "review_required",
                    "sfx": {
                        "visual_detector": "sfx_text_detector",
                        "source_text": "",
                        "adapted_text": "",
                        "inpaint_allowed": False,
                    },
                },
                translated="",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertNotIn("style_evidence", layer)
        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["render_policy"], "review_required")

    def test_build_text_layer_skips_source_style_scan_for_small_review_sfx_visual(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((180, 120, 3), dtype=np.uint8)

        def fail_extract(_crop):
            raise AssertionError("style extractor should not run for non-promoted SFX visuals")

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fail_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "sfx_visual_ornament",
                    "text": "",
                    "bbox": [30, 20, 70, 150],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "route_action": "review_required",
                    "render_policy": "review_required",
                    "sfx": {
                        "visual_detector": "sfx_visual",
                        "source_text": "",
                        "adapted_text": "",
                        "inpaint_allowed": False,
                    },
                },
                translated="",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertNotIn("style_evidence", layer)
        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["render_policy"], "review_required")

    def test_build_text_layer_skips_source_style_scan_for_low_confidence_primary_ocr(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((120, 260, 3), dtype=np.uint8)

        def fail_extract(_crop):
            raise AssertionError("style extractor should not run for low-confidence OCR text")

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fail_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "ocr_low_conf",
                    "text": "ALCHEMY, THIS IS TRULY AN INSANE ABILITY.",
                    "bbox": [20, 20, 240, 90],
                    "content_class": "text",
                    "tipo": "fala",
                    "confidence": 0.52,
                    "route_action": "translate_inpaint_render",
                },
                translated="ALQUIMIA, ISSO E UMA HABILIDADE INSANA.",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertNotIn("style_evidence", layer)
        self.assertEqual(layer["style_origin"], "auto")

    def test_build_text_layer_extracts_source_style_for_confident_promoted_sfx_detector(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((160, 320, 3), dtype=np.uint8)
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#010100",
            "text_color_confidence": 0.96,
            "stroke_color": "#FFFFFE",
            "stroke_width_px": 2,
            "stroke_confidence": 0.92,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.74,
        }
        crop_shapes = []

        def fake_extract(crop):
            crop_shapes.append(tuple(crop.shape))
            return evidence

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fake_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "sfx_promoted_confident",
                    "text": "BOOM",
                    "bbox": [30, 30, 280, 120],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "confidence": 0.82,
                    "route_action": "translate_sfx_inpaint_render",
                    "render_policy": "sfx_style",
                    "sfx_promotion_score": 0.74,
                    "sfx": {
                        "visual_detector": "sfx_visual",
                        "visual_promotion": True,
                        "visual_confidence": 0.82,
                        "source_text": "BOOM",
                        "adapted_text": "BOOM",
                    },
                },
                translated="BOOM",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertEqual(crop_shapes, [(90, 250, 3)])
        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["fonte"], "KOMIKAX_.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "#FFFFFE")

    def test_build_text_layer_skips_source_style_for_low_confidence_promoted_sfx_detector(self) -> None:
        import numpy as np

        source_image_rgb = np.zeros((160, 320, 3), dtype=np.uint8)

        def fail_extract(_crop):
            raise AssertionError("style extractor should not run for low-confidence SFX detector")

        with patch("typesetter.style_extractor.extract_text_style_evidence", side_effect=fail_extract):
            layer = main.build_text_layer(
                page_number=1,
                layer_index=0,
                ocr_text={
                    "id": "sfx_promoted_low_conf",
                    "text": "BOOM",
                    "bbox": [30, 30, 280, 120],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "confidence": 0.52,
                    "route_action": "translate_sfx_inpaint_render",
                    "render_policy": "sfx_style",
                    "sfx_promotion_score": 0.55,
                    "sfx": {
                        "visual_detector": "sfx_visual",
                        "visual_promotion": True,
                        "visual_confidence": 0.52,
                        "source_text": "BOOM",
                        "adapted_text": "BOOM",
                    },
                },
                translated="BOOM",
                corpus_visual_benchmark={},
                corpus_textual_benchmark={},
                source_image_rgb=source_image_rgb,
            )

        self.assertNotIn("style_evidence", layer)
        self.assertEqual(layer["style_origin"], "auto")

    def test_build_text_layer_does_not_apply_style_evidence_for_low_confidence_primary_ocr(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.96,
            "stroke_color": "#101010",
            "stroke_width_px": 3,
            "stroke_confidence": 0.93,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.82,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "ocr_low_conf_with_style",
                "text": "ALCHEMY, THIS IS TRULY AN INSANE ABILITY.",
                "bbox": [20, 20, 240, 90],
                "content_class": "text",
                "tipo": "fala",
                "confidence": 0.52,
                "route_action": "translate_inpaint_render",
                "style_evidence": evidence,
            },
            translated="ALQUIMIA, ISSO E UMA HABILIDADE INSANA.",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_does_not_apply_style_evidence_for_low_confidence_sfx_detector(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.96,
            "stroke_color": "#101010",
            "stroke_width_px": 3,
            "stroke_confidence": 0.93,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.82,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_low_conf_with_style",
                "text": "BOOM",
                "bbox": [20, 20, 240, 90],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "confidence": 0.52,
                "route_action": "translate_sfx_inpaint_render",
                "render_policy": "sfx_style",
                "sfx_promotion_score": 0.55,
                "style_evidence": evidence,
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "visual_promotion": True,
                    "visual_confidence": 0.52,
                    "source_text": "BOOM",
                    "adapted_text": "BOOM",
                },
            },
            translated="BOOM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_uses_conservative_style_effect_defaults_from_high_confidence_evidence(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "shadow": True,
            "shadow_confidence": 0.7,
            "glow": True,
            "glow_confidence": 0.7,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "BOOM",
                "bbox": [0, 0, 100, 100],
                "tipo": "sfx",
                "confidence": 0.92,
                "background_rgb": [30, 30, 30],
                "style_evidence": evidence,
            },
            translated="BUM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_confidence"], 0.7)
        self.assertTrue(layer["estilo"]["sombra"])
        self.assertEqual(layer["estilo"]["sombra_cor"], "#000000")
        self.assertEqual(layer["estilo"]["sombra_offset"], [2, 2])
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#FFFFFF")
        self.assertEqual(layer["estilo"]["glow_px"], 2)

    def test_build_text_layer_applies_high_confidence_curved_style_evidence(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#000000",
            "text_color_confidence": 0.5,
            "curved": True,
            "curve_direction": "arc_up",
            "curve_amount": 0.36,
            "curve_confidence": 0.82,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "OLA TUDO BEM",
                "bbox": [0, 0, 180, 90],
                "tipo": "sfx",
                "confidence": 0.92,
                "background_rgb": [255, 255, 255],
                "style_evidence": evidence,
            },
            translated="OLA TUDO BEM",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["style_confidence"], 0.82)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertTrue(layer["estilo"]["curva"])
        self.assertEqual(layer["estilo"]["curva_direcao"], "arc_up")
        self.assertEqual(layer["estilo"]["curva_intensidade"], 0.36)

    def test_build_text_layer_keeps_low_confidence_style_evidence_but_uses_auto_style(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "text_color": "#FFFFFF",
            "text_color_confidence": 0.42,
            "stroke_color": "#000000",
            "stroke_width_px": 3,
            "stroke_confidence": 0.31,
            "font_name": "KOMIKAX_.ttf",
            "font_confidence": 0.5,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "HELLO",
                "bbox": [0, 0, 100, 100],
                "tipo": "fala",
                "confidence": 0.92,
                "background_rgb": [250, 250, 250],
                "style_evidence": evidence,
            },
            translated="OLA",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.5)
        self.assertEqual(layer["style_source"], "pixel_analysis")
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertIs(layer["style"], layer["estilo"])
        self.assertEqual(layer["estilo"]["style_origin"], "auto")
        self.assertEqual(layer["estilo"]["fonte"], "ComicNeue-Bold.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#000000")
        self.assertEqual(layer["estilo"]["contorno"], "")
        self.assertEqual(layer["estilo"]["contorno_px"], 0)

    def test_build_text_layer_keeps_low_confidence_style_effect_evidence_without_applying_effects(self) -> None:
        evidence = {
            "source": "pixel_analysis",
            "shadow": True,
            "shadow_confidence": 0.69,
            "shadow_color": "#111111",
            "shadow_offset": [3, 4],
            "glow": True,
            "glow_confidence": 0.69,
            "glow_color": "#AAFFFF",
            "glow_px": 5,
        }

        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "tl_001_001",
                "text": "HELLO",
                "bbox": [0, 0, 100, 100],
                "tipo": "fala",
                "confidence": 0.92,
                "background_rgb": [250, 250, 250],
                "style_evidence": evidence,
            },
            translated="OLA",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["style_confidence"], 0.69)
        self.assertEqual(layer["style_evidence"], evidence)
        self.assertFalse(layer["estilo"]["sombra"])
        self.assertEqual(layer["estilo"]["sombra_cor"], "")
        self.assertEqual(layer["estilo"]["sombra_offset"], [0, 0])
        self.assertFalse(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "")
        self.assertEqual(layer["estilo"]["glow_px"], 0)

    def test_build_text_layer_persists_dark_panel_glow_fallback_for_rejected_mask(self) -> None:
        layer = main.build_text_layer(
            page_number=6,
            layer_index=0,
            ocr_text={
                "id": "ocr_002",
                "text": "the Devil Knight!",
                "bbox": [178, 14378, 371, 14413],
                "text_pixel_bbox": [178, 14378, 371, 14413],
                "tipo": "text",
                "confidence": 0.56,
                "background_rgb": [87, 78, 45],
                "bubble_mask_source": "derived_white_crop_rejected",
                "layout_safe_reason": "debug_derived_bubble_mask_rejected",
                "qa_flags": ["debug_derived_bubble_mask_rejected"],
            },
            translated="O cavaleiro do diabo!",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(layer["estilo"]["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(layer["estilo"]["cor"], "#FFFFFF")
        self.assertEqual(layer["estilo"]["contorno"], "#061D26")
        self.assertGreaterEqual(layer["estilo"]["contorno_px"], 1)
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#67D8FF")
        self.assertGreaterEqual(layer["estilo"]["glow_px"], 3)

    def test_build_text_layer_uses_original_dark_panel_effect_colors(self) -> None:
        layer = main.build_text_layer(
            page_number=6,
            layer_index=0,
            ocr_text={
                "id": "ocr_002",
                "text": "Quest Introduction!",
                "bbox": [118, 186, 390, 304],
                "text_pixel_bbox": [118, 186, 390, 304],
                "tipo": "text",
                "confidence": 0.56,
                "background_rgb": [2, 2, 2],
                "bubble_mask_source": "image_dark_panel_mask",
                "bubble_mask_bbox": [61, 122, 443, 356],
                "dark_panel_effect_colors": {
                    "color_sample_space": "original_image",
                    "panel_fill_rgb": [3, 3, 3],
                    "border_rgb": [160, 155, 142],
                    "panel_glow_rgb": [75, 61, 24],
                    "text_fill_rgb": [252, 250, 236],
                    "text_glow_rgb": [83, 71, 39],
                    "bad_negative_text_glow_rgb": [186, 197, 236],
                },
            },
            translated="Introducao da missao!",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(layer["style_source"], "original_dark_panel_effect_colors")
        self.assertEqual(layer["estilo"]["cor"], "#FCFAEC")
        self.assertEqual(layer["estilo"]["contorno"], "#A09B8E")
        self.assertEqual(layer["estilo"]["glow_cor"], "#534727")
        self.assertNotEqual(layer["estilo"]["glow_cor"], "#67D8FF")
        self.assertEqual(layer["dark_panel_effect_colors"]["bad_negative_text_glow_rgb"], [186, 197, 236])
        self.assertIn("original_dark_panel_effect_colors", layer.get("qa_flags") or [])

    def test_build_text_layer_does_not_apply_dark_panel_glow_fallback_to_white_balloon(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "ocr_001",
                "text": "WAIT",
                "bbox": [10, 20, 140, 70],
                "tipo": "text",
                "confidence": 0.56,
                "background_rgb": [255, 255, 255],
                "layout_profile": "white_balloon",
                "bubble_mask_source": "derived_white_crop_rejected",
                "layout_safe_reason": "debug_derived_bubble_mask_rejected",
                "qa_flags": ["debug_derived_bubble_mask_rejected"],
            },
            translated="ESPERE",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["estilo"]["cor"], "#000000")
        self.assertFalse(layer["estilo"]["glow"])

    def test_build_text_layer_does_not_apply_visual_card_style_to_contour_balloon(self) -> None:
        layer = main.build_text_layer(
            page_number=2,
            layer_index=0,
            ocr_text={
                "id": "ocr_001",
                "text": "DON'T HIT MY MOM!",
                "bbox": [268, 5829, 446, 5902],
                "text_pixel_bbox": [268, 5829, 446, 5902],
                "tipo": "text",
                "confidence": 0.95,
                "background_rgb": [92, 154, 224],
                "bubble_mask_source": "image_contour_bubble_mask",
                "qa_flags": ["safe_text_box_recomputed"],
            },
            translated="Não bata na minha mãe!",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "auto")
        self.assertEqual(layer["estilo"]["cor"], "#000000")
        self.assertFalse(layer["estilo"]["glow"])
        self.assertNotIn("visual_card_style_fallback", layer.get("qa_flags") or [])

    def test_final_project_applies_dark_panel_glow_after_debug_mask_flags(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "translated": "O cavaleiro do diabo!",
                            "background_rgb": [87, 78, 45],
                            "bubble_mask_source": "derived_white_crop_rejected",
                            "layout_safe_reason": "debug_derived_bubble_mask_rejected",
                            "qa_flags": ["debug_derived_bubble_mask_rejected"],
                            "style_origin": "auto",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FFFFFF",
                                "contorno": "",
                                "contorno_px": 0,
                                "glow": False,
                                "glow_cor": "",
                                "glow_px": 0,
                            },
                        }
                    ]
                }
            ]
        }

        applied = main._apply_dark_panel_glow_project_styles(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(applied, 1)
        self.assertEqual(layer["style_origin"], "auto_dark_panel_glow")
        self.assertEqual(layer["estilo"]["style_origin"], "auto_dark_panel_glow")
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#67D8FF")
        self.assertIn("auto_dark_panel_glow_fallback", layer["qa_flags"])

    def test_build_text_layer_uses_visual_card_style_for_colored_status_panel(self) -> None:
        layer = main.build_text_layer(
            page_number=6,
            layer_index=0,
            ocr_text={
                "id": "ocr_002",
                "text": "Synching is complete.",
                "bbox": [442, 96, 570, 160],
                "text_pixel_bbox": [455, 106, 559, 166],
                "tipo": "text",
                "confidence": 0.95,
                "background_rgb": [92, 154, 224],
                "bubble_mask_source": "image_white_bubble_mask",
                "qa_flags": ["safe_text_box_recomputed"],
            },
            translated="A sincronização foi concluída.",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["style_origin"], "inferred_visual_card")
        self.assertEqual(layer["estilo"]["style_origin"], "inferred_visual_card")
        self.assertEqual(layer["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#EBFFFF")
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#EBFFFF")
        self.assertGreaterEqual(layer["estilo"]["glow_px"], 2)
        self.assertIn("visual_card_style_fallback", layer["qa_flags"])

    def test_final_project_applies_visual_card_style_from_render_metrics(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "translated": "A sincronização foi concluída.",
                            "bubble_mask_source": "image_white_bubble_mask",
                            "qa_flags": ["safe_text_box_recomputed", "render_on_art_suspected"],
                            "style_origin": "auto",
                            "qa_metrics": {
                                "render_background_luma": 176.79,
                                "render_background_luma_std": 2.05,
                            },
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#000000",
                                "contorno": "",
                                "contorno_px": 0,
                                "glow": False,
                                "glow_cor": "",
                                "glow_px": 0,
                            },
                        }
                    ]
                }
            ]
        }

        applied = main._apply_dark_panel_glow_project_styles(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(applied, 1)
        self.assertEqual(layer["style_origin"], "inferred_visual_card")
        self.assertEqual(layer["estilo"]["style_origin"], "inferred_visual_card")
        self.assertEqual(layer["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#EBFFFF")
        self.assertTrue(layer["estilo"]["glow"])
        self.assertEqual(layer["estilo"]["glow_cor"], "#EBFFFF")
        self.assertIn("visual_card_style_fallback", layer["qa_flags"])

    def test_final_project_uses_visual_card_font_for_source_detected_status_panel(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "translated": "O anfitrião recebeu o título,",
                            "bubble_mask_source": "image_white_bubble_mask",
                            "qa_flags": ["safe_text_box_recomputed", "render_on_art_suspected"],
                            "style_origin": "source_detected",
                            "qa_metrics": {
                                "render_background_luma": 174.22,
                                "render_background_luma_std": 12.56,
                            },
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#EBFFFF",
                                "glow": True,
                                "glow_cor": "#EBFFFF",
                                "glow_px": 2,
                                "style_origin": "source_detected",
                                "style_confidence": 0.9,
                                "style_source": "pixel_analysis",
                            },
                        }
                    ]
                }
            ]
        }

        applied = main._apply_dark_panel_glow_project_styles(project)
        layer = project["paginas"][0]["text_layers"][0]

        self.assertEqual(applied, 1)
        self.assertEqual(layer["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["style_origin"], "source_detected")
        self.assertEqual(layer["estilo"]["style_source"], "pixel_analysis")
        self.assertEqual(layer["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(layer["estilo"]["cor"], "#EBFFFF")
        self.assertTrue(layer["estilo"]["glow"])
        self.assertIn("visual_card_font_fallback", layer["qa_flags"])

    def test_final_project_groups_similar_dark_panel_visual_styles(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "dark_criteria",
                            "translated": "OS CRITERIOS DA MISSAO",
                            "background_rgb": [8, 15, 19],
                            "bubble_mask_source": "image_dark_panel_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [3, 7, 10],
                                "text_fill_rgb": [244, 252, 255],
                                "text_glow_rgb": [84, 214, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "KOMIKAX_.ttf",
                                "cor": "#F4FCFF",
                                "contorno": "#061D26",
                                "contorno_px": 1,
                                "glow": True,
                                "glow_cor": "#54D6FF",
                                "glow_px": 3,
                                "tamanho": 28,
                                "style_origin": "auto_dark_panel_glow",
                                "style_source": "original_dark_panel_effect_colors",
                            },
                            "text_pixel_bbox": [20, 20, 190, 78],
                            "qa_flags": ["auto_dark_panel_glow_fallback", "original_dark_panel_effect_colors"],
                        },
                        {
                            "id": "dark_reward",
                            "translated": "A RECOMPENSA DA MISSAO",
                            "background_rgb": [12, 18, 22],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [4, 8, 12],
                                "text_fill_rgb": [250, 255, 255],
                                "text_glow_rgb": [91, 221, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FFFFFF",
                                "contorno": "#061D26",
                                "contorno_px": 1,
                                "glow": True,
                                "glow_cor": "#5BDDFF",
                                "glow_px": 3,
                                "tamanho": 16,
                                "style_origin": "auto_dark_panel_glow",
                                "style_source": "original_dark_panel_effect_colors",
                            },
                            "text_pixel_bbox": [22, 28, 150, 68],
                            "qa_flags": ["auto_dark_panel_glow_fallback", "original_dark_panel_effect_colors"],
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        first, second = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["layers"], 2)
        self.assertEqual(result["condensed_font_groups"], 1)
        self.assertEqual(first["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(second["estilo"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
        self.assertEqual(first["estilo"]["cor"], second["estilo"]["cor"])
        self.assertEqual(first["estilo"]["glow_cor"], second["estilo"]["glow_cor"])
        self.assertEqual(first["style_group_id"], second["style_group_id"])
        self.assertEqual(first["estilo"]["tamanho"], 28)
        self.assertEqual(second["estilo"]["tamanho"], 16)
        self.assertIn("dark_panel_style_grouped", first["qa_flags"])
        self.assertIn("dark_panel_condensed_group_font", second["qa_flags"])

    def test_final_project_dark_bubble_group_does_not_copy_komikax_to_long_text(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "short_decorative",
                            "translated": "MISSAO",
                            "background_rgb": [8, 15, 19],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [3, 7, 10],
                                "text_fill_rgb": [244, 252, 255],
                                "text_glow_rgb": [84, 214, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "KOMIKAX_.ttf",
                                "cor": "#F4FCFF",
                                "contorno": "#061D26",
                                "contorno_px": 1,
                                "glow": True,
                                "glow_cor": "#54D6FF",
                                "glow_px": 3,
                                "tamanho": 28,
                            },
                            "text_pixel_bbox": [20, 20, 190, 78],
                        },
                        {
                            "id": "long_dialogue",
                            "translated": "Critérios de conclusão da missão: estabeleça um submundo de nível 1.",
                            "background_rgb": [8, 15, 19],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [3, 7, 10],
                                "text_fill_rgb": [244, 252, 255],
                                "text_glow_rgb": [84, 214, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FFFFFF",
                                "contorno": "#061D26",
                                "contorno_px": 1,
                                "glow": True,
                                "glow_cor": "#54D6FF",
                                "glow_px": 3,
                                "tamanho": 25,
                            },
                            "text_pixel_bbox": [22, 28, 150, 68],
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        short, long = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 1)
        self.assertNotEqual(long["estilo"]["fonte"], "KOMIKAX_.ttf")
        self.assertIn(long["estilo"]["fonte"], {"ComicNeue-Bold.ttf", "LeagueGothic-Regular-VariableFont_wdth.ttf"})
        self.assertEqual(long["estilo"]["cor"], short["estilo"]["cor"])
        self.assertEqual(long["estilo"]["glow_cor"], short["estilo"]["glow_cor"])
        self.assertIn("dark_panel_style_grouped", long["qa_flags"])

    def test_final_project_dark_bubble_group_caps_thick_source_outline(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "thick_outline_source",
                            "translated": "Quest completion criteria",
                            "background_rgb": [1, 1, 1],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [1, 1, 1],
                                "text_fill_rgb": [253, 252, 246],
                                "text_glow_rgb": [103, 216, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FDFCF6",
                                "contorno": "#4F4E4B",
                                "contorno_px": 8,
                                "glow": True,
                                "glow_cor": "#67D8FF",
                                "glow_px": 3,
                                "tamanho": 27,
                            },
                            "text_pixel_bbox": [82, 113, 367, 219],
                        },
                        {
                            "id": "same_dark_bubble_target",
                            "translated": "Critérios de conclusão da missão: estabeleça um submundo de nível 1.",
                            "background_rgb": [1, 1, 1],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [1, 1, 1],
                                "text_fill_rgb": [253, 252, 246],
                                "text_glow_rgb": [103, 216, 255],
                            },
                            "style_origin": "auto_dark_panel_glow",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#FFFFFF",
                                "contorno": "#061D26",
                                "contorno_px": 1,
                                "glow": True,
                                "glow_cor": "#67D8FF",
                                "glow_px": 3,
                                "tamanho": 25,
                            },
                            "text_pixel_bbox": [84, 116, 360, 224],
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        source, target = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 1)
        self.assertEqual(source["estilo"]["contorno_px"], 1)
        self.assertEqual(target["estilo"]["contorno_px"], 1)
        self.assertEqual(target["estilo"]["glow_cor"], "#67D8FF")
        self.assertIn("dark_panel_group_outline_capped", source["qa_flags"])
        self.assertIn("dark_panel_style_grouped", target["qa_flags"])

    def test_final_project_dark_panel_groups_ignore_white_balloons_and_different_effects(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "white_balloon",
                            "translated": "EU REALMENTE NAO TENHO DINHEIRO",
                            "background_rgb": [248, 248, 248],
                            "bubble_mask_source": "image_white_bubble_mask",
                            "block_profile": "white_balloon",
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#000000",
                                "glow": False,
                                "glow_cor": "",
                                "glow_px": 0,
                            },
                        },
                        {
                            "id": "dark_blue",
                            "translated": "THE EPISODE STARTS",
                            "background_rgb": [6, 12, 16],
                            "bubble_mask_source": "image_dark_panel_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [2, 4, 8],
                                "text_fill_rgb": [242, 252, 255],
                                "text_glow_rgb": [72, 204, 255],
                            },
                            "estilo": {"fonte": "ComicNeue-Bold.ttf", "cor": "#FFFFFF", "glow": True, "glow_cor": "#48CCFF"},
                        },
                        {
                            "id": "dark_warm",
                            "translated": "WARNING",
                            "background_rgb": [8, 5, 4],
                            "bubble_mask_source": "image_dark_panel_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [5, 3, 2],
                                "text_fill_rgb": [255, 232, 196],
                                "text_glow_rgb": [255, 124, 44],
                            },
                            "estilo": {"fonte": "KOMIKAX_.ttf", "cor": "#FFE8C4", "glow": True, "glow_cor": "#FF7C2C"},
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        white, dark_blue, dark_warm = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 0)
        self.assertNotIn("style_group_id", white)
        self.assertNotIn("style_group_id", dark_blue)
        self.assertNotIn("style_group_id", dark_warm)
        self.assertEqual(white["estilo"]["fonte"], "ComicNeue-Bold.ttf")

    def test_final_project_dark_panel_groups_ignore_false_dark_white_context(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "false_dark_white_balloon",
                            "translated": "O QUE VOCE QUER DIZER COM SUBESPACO?",
                            "background_rgb": [247, 247, 247],
                            "bubble_mask_source": "image_white_bubble_mask",
                            "block_profile": "fala",
                            "qa_flags": [
                                "dark_bubble_oval_reocr",
                                "dark_bubble_ellipse_bbox_mask",
                                "dark_bubble_visual_glyph_mask_replaced_geometry",
                                "trusted_dark_visual_capacity_target",
                            ],
                            "estilo": {
                                "fonte": "ComicNeue-Bold.ttf",
                                "cor": "#000000",
                                "glow": False,
                                "glow_cor": "",
                                "glow_px": 0,
                            },
                        },
                        {
                            "id": "real_dark_panel",
                            "translated": "A MISSAO PRINCIPAL SERA MOSTRADA EM BREVE",
                            "background_rgb": [5, 10, 14],
                            "bubble_mask_source": "image_dark_panel_mask",
                            "block_profile": "dark_panel",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [2, 4, 8],
                                "text_fill_rgb": [242, 252, 255],
                                "text_glow_rgb": [72, 204, 255],
                            },
                            "estilo": {"fonte": "ComicNeue-Bold.ttf", "cor": "#FFFFFF", "glow": True, "glow_cor": "#48CCFF"},
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        false_white, real_dark = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 0)
        self.assertNotIn("style_group_id", false_white)
        self.assertNotIn("dark_panel_style_grouped", false_white.get("qa_flags", []))
        self.assertNotIn("style_group_id", real_dark)

    def test_final_project_dark_panel_groups_ignore_promoted_light_dark_bubble(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "promoted_light_dark_bubble",
                            "translated": "O QUE VOCE QUER DIZER COM SUBESPACO?",
                            "background_rgb": [245, 245, 245],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "white_balloon",
                            "qa_flags": [
                                "false_light_bubble_dark_fill_blocked",
                                "false_light_dark_bubble_promoted_to_white",
                                "dark_panel_style_grouped",
                            ],
                            "estilo": {"fonte": "ComicNeue-Bold.ttf", "cor": "#000000", "glow": False},
                        },
                        {
                            "id": "real_dark_bubble",
                            "translated": "A MISSAO RECOMPENSA E",
                            "background_rgb": [5, 8, 10],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "block_profile": "dark_bubble",
                            "dark_panel_effect_colors": {
                                "panel_fill_rgb": [0, 0, 0],
                                "text_fill_rgb": [245, 250, 255],
                                "text_glow_rgb": [60, 190, 255],
                            },
                            "estilo": {"fonte": "LeagueGothic-Regular-VariableFont_wdth.ttf", "cor": "#FFFFFF", "glow": True},
                        },
                    ]
                }
            ]
        }

        result = main._apply_dark_panel_style_groups(project)
        promoted, real_dark = project["paginas"][0]["text_layers"]

        self.assertEqual(result["groups"], 0)
        self.assertNotIn("style_group_id", promoted)
        self.assertNotIn("style_group_id", real_dark)

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

    def test_build_text_layer_enriches_sfx_route_without_overwriting_policies(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "text": "\ucff5",
                "bbox": [10, 10, 180, 80],
                "content_class": "sfx",
                "script": "hangul",
                "translate_policy": "adapt_sfx",
                "render_policy": "sfx_style",
                "route_action": "translate_sfx_inpaint_render",
                "route_reason": "hangul_sfx_candidate",
            },
            translated="\ucff5",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["content_class"], "sfx")
        self.assertEqual(layer["script"], "hangul")
        self.assertEqual(layer["translate_policy"], "adapt_sfx")
        self.assertEqual(layer["render_policy"], "sfx_style")
        self.assertEqual(layer["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(layer["translated"], "TUM")
        self.assertEqual(layer["traduzido"], "TUM")
        self.assertEqual(layer["sfx"]["source_text"], "\ucff5")
        self.assertEqual(layer["sfx"]["adapted_text"], "TUM")
        self.assertEqual(layer["sfx"]["translation_mode"], "onomatopoeia_adaptation")
        self.assertFalse(layer["sfx"]["inpaint_allowed"])

    def test_build_text_layer_mixed_dialogue_safe_and_unsafe_sfx_contract(self) -> None:
        dialogue = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "dialogue",
                "text": "HELLO",
                "bbox": [10, 10, 110, 60],
                "tipo": "fala",
            },
            translated="OLA",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )
        safe_sfx = main.build_text_layer(
            page_number=1,
            layer_index=1,
            ocr_text={
                "id": "safe_sfx",
                "text": "\ucff5",
                "bbox": [120, 10, 220, 90],
                "content_class": "sfx",
                "route_action": "translate_sfx_inpaint_render",
                "sfx": {"inpaint_allowed": True},
            },
            translated="\ucff5",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )
        unsafe_sfx = main.build_text_layer(
            page_number=1,
            layer_index=2,
            ocr_text={
                "id": "unsafe_sfx",
                "text": "\ucff5",
                "bbox": [230, 10, 330, 90],
                "content_class": "sfx",
                "route_action": "translate_sfx_inpaint_render",
                "sfx": {
                    "inpaint_allowed": False,
                    "qa_flags": ["complex_background"],
                },
            },
            translated="\ucff5",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(dialogue["route_action"], "translate_inpaint_render")
        self.assertEqual(dialogue["translated"], "OLA")
        self.assertEqual(safe_sfx["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(safe_sfx["translated"], "TUM")
        self.assertTrue(safe_sfx["sfx"]["inpaint_allowed"])
        self.assertEqual(unsafe_sfx["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(unsafe_sfx["sfx"]["adapted_text"], "TUM")
        self.assertFalse(unsafe_sfx["sfx"]["inpaint_allowed"])
        self.assertIn("complex_background", unsafe_sfx["sfx"]["qa_flags"])

    def test_promote_sfx_visual_candidate_keeps_unknown_script_review_only(self) -> None:
        page_result = {
            "texts": [{"id": "dialogue", "bbox": [10, 10, 80, 40], "text": "HELLO"}],
            "_sfx_visual_candidates": [
                {
                    "id": "sfx_visual_001",
                    "bbox": [120, 20, 220, 110],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "route_action": "review_required",
                    "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                    "sfx": {
                        "visual_detector": "sfx_visual",
                        "visual_confidence": 0.71,
                        "inpaint_allowed": False,
                        "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                    },
                }
            ],
        }

        promoted = main._promote_sfx_visual_candidates(page_result)

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["route_action"], "review_required")
        self.assertEqual(promoted[0]["script"], "visual_unknown")
        self.assertFalse(promoted[0]["sfx"]["inpaint_allowed"])

    def test_promote_sfx_visual_candidate_with_strong_visual_evidence_enters_sfx_route(self) -> None:
        image = np.full((140, 240, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (126, 38), (140, 96), (15, 15, 15), -1)
        cv2.rectangle(image, (126, 82), (178, 96), (15, 15, 15), -1)
        cv2.rectangle(image, (186, 38), (202, 102), (15, 15, 15), -1)
        page_result = {
            "texts": [{"id": "dialogue", "bbox": [10, 10, 80, 40], "text": "HELLO"}],
            "_cached_image_rgb": image,
            "_sfx_visual_candidates": [
                {
                    "id": "sfx_visual_001",
                    "bbox": [110, 20, 220, 120],
                    "content_class": "sfx",
                    "tipo": "sfx",
                    "detector": "sfx_visual",
                    "confidence": 0.82,
                    "route_action": "review_required",
                    "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                    "sfx_ocr": {"status": "no_confident_cjk"},
                    "sfx": {
                        "visual_detector": "sfx_visual",
                        "visual_source": "local_contrast",
                        "visual_confidence": 0.82,
                        "inpaint_allowed": False,
                        "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                    },
                }
            ],
        }

        promoted = main._promote_sfx_visual_candidates(page_result, sfx_ocr_recognizer=lambda _crop, _lang: [])

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(promoted[0]["script"], "visual_unknown")
        self.assertEqual(promoted[0]["translate_policy"], "review")
        self.assertEqual(promoted[0]["render_policy"], "sfx_style")
        self.assertTrue(promoted[0]["sfx"]["visual_promotion"])
        self.assertFalse(promoted[0]["sfx"]["inpaint_allowed"])
        self.assertIn("sfx_inpaint_gate", promoted[0])
        self.assertIn("sfx_text_unknown", promoted[0]["qa_flags"])

    def test_drop_suppressed_ocr_texts_uses_sfx_visual_candidates_before_promotion(self) -> None:
        image = np.full((140, 240, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (126, 38), (140, 96), (15, 15, 15), -1)
        cv2.rectangle(image, (126, 82), (178, 96), (15, 15, 15), -1)
        cv2.rectangle(image, (186, 38), (202, 102), (15, 15, 15), -1)
        texts = [
            {
                "id": "ocr_false_sfx",
                "bbox": [120, 20, 220, 110],
                "text": "XEV",
                "route_action": "translate_inpaint_render",
            }
        ]
        sfx_candidates = [
            {
                "id": "sfx_visual_001",
                "bbox": [110, 20, 220, 120],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "confidence": 0.82,
                "qa_flags": ["sfx_visual_candidate"],
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "visual_confidence": 0.82,
                    "visual_source": "local_contrast",
                    "inpaint_allowed": False,
                },
            }
        ]
        page_result = {
            "texts": texts,
            "_cached_image_rgb": image,
            "_sfx_visual_candidates": sfx_candidates,
        }

        filtered = main._drop_suppressed_ocr_texts(texts, "en", sfx_candidates=sfx_candidates)
        promoted = main._promote_sfx_visual_candidates(
            page_result,
            existing_texts=filtered,
            sfx_ocr_recognizer=lambda _crop, _lang: [],
        )

        self.assertEqual(filtered, [])
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["route_action"], "translate_sfx_inpaint_render")

    def test_build_text_layer_preserves_visual_sfx_unknown_review_only(self) -> None:
        layer = main.build_text_layer(
            page_number=1,
            layer_index=0,
            ocr_text={
                "id": "sfx_visual_001",
                "text": "",
                "bbox": [20, 20, 120, 120],
                "content_class": "sfx",
                "tipo": "sfx",
                "detector": "sfx_visual",
                "route_action": "review_required",
                "translate_policy": "review",
                "render_policy": "review_required",
                "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                "sfx": {
                    "visual_detector": "sfx_visual",
                    "visual_confidence": 0.68,
                    "inpaint_allowed": False,
                    "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
                },
            },
            translated="",
            corpus_visual_benchmark={},
            corpus_textual_benchmark={},
        )

        self.assertEqual(layer["route_action"], "review_required")
        self.assertEqual(layer["script"], "unknown")
        self.assertEqual(layer["render_policy"], "review_required")
        self.assertIn("sfx_script_unknown", layer["qa_flags"])

    def test_promote_sfx_visual_candidate_with_hangul_uses_sfx_route(self) -> None:
        promoted = main._promote_sfx_visual_candidates(
            {
                "texts": [],
                "_sfx_visual_candidates": [
                    {
                        "id": "sfx_visual_001",
                        "bbox": [20, 20, 120, 120],
                        "content_class": "sfx",
                        "tipo": "sfx",
                        "recognized_text": "\ucff5",
                        "sfx": {"visual_detector": "sfx_visual", "inpaint_allowed": False},
                    }
                ],
            }
        )

        self.assertEqual(promoted[0]["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(promoted[0]["script"], "hangul")
        self.assertEqual(promoted[0]["sfx"]["source_text"], "\ucff5")

    def test_promote_sfx_visual_candidate_runs_cjk_ocr_probe_from_cached_image(self) -> None:
        image = np.full((80, 100, 3), 240, dtype=np.uint8)
        calls = []

        def recognizer(crop, lang):
            calls.append(lang)
            return [{"text": "\ucff5", "confidence": 0.82}] if lang == "ko" else []

        promoted = main._promote_sfx_visual_candidates(
            {
                "texts": [],
                "_cached_image_rgb": image,
                "_sfx_visual_candidates": [
                    {
                        "id": "sfx_visual_001",
                        "bbox": [20, 20, 60, 60],
                        "content_class": "sfx",
                        "tipo": "sfx",
                        "sfx": {"visual_detector": "sfx_visual", "inpaint_allowed": False},
                    }
                ],
            },
            sfx_ocr_recognizer=recognizer,
        )

        self.assertIn("ko", calls)
        self.assertEqual(promoted[0]["route_action"], "translate_sfx_inpaint_render")
        self.assertEqual(promoted[0]["recognized_text"], "\ucff5")
        self.assertEqual(promoted[0]["sfx_ocr"]["status"], "recognized")

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

    def test_hydrate_project_render_metadata_uses_raw_render_plan_when_candidate_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_003@page_014_band_116",
                        "text_id": "ocr_003",
                        "band_id": "page_014_band_116",
                        "page_id": "page_014",
                        "target_bbox": [365, 284, 533, 374],
                        "balloon_bbox": [365, 319, 533, 374],
                        "render_bbox": [402, 315, 494, 333],
                        "safe_text_box": [400, 296, 496, 352],
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
                        "trace_id": "ocr_003@page_014_band_116",
                        "text_id": "ocr_003",
                        "band_id": "page_014_band_116",
                        "page_id": "page_014",
                        "bboxes": {"bbox": {"value": [365, 4907, 533, 4962]}},
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
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_014_band_116",
                                "band_id": "page_014_band_116",
                                "page_id": "page_014",
                                "route_action": "translate_inpaint_render",
                                "translated": "LENTO...",
                                "bbox": [365, 4907, 533, 4962],
                                "balloon_bbox": [365, 4907, 533, 4962],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["raw_candidate_count"], 1)
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [402, 4903, 494, 4921])
        self.assertEqual(layer["safe_text_box"], [400, 4884, 496, 4940])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_clamp_project_render_geometry_to_page_limits_cross_page_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            work_dir = Path(tmp)
            (work_dir / "originals").mkdir(parents=True)
            Image.new("RGB", (200, 300), "white").save(work_dir / "originals" / "001.jpg")
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "arquivo_original": "originals/001.jpg",
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "target_bbox": [0, 240, 200, 360],
                                "safe_text_box": [20, 260, 180, 340],
                                "_debug_safe_text_box": [20, 260, 180, 340],
                                "render_bbox": [40, 280, 160, 330],
                                "balloon_bbox": [0, 240, 200, 360],
                                "bubble_mask_bbox": [0, 240, 200, 360],
                            }
                        ],
                    }
                ],
            }

            audit = main._clamp_project_render_geometry_to_page(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["bbox_clamped_count"], 6)
        self.assertEqual(layer["target_bbox"], [0, 240, 200, 300])
        self.assertEqual(layer["safe_text_box"], [20, 260, 180, 300])
        self.assertEqual(layer["render_bbox"], [40, 280, 160, 300])

    def test_repair_project_real_bubble_body_safe_area_uses_balloon_body(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "translated": "HYUNG-NIM! ENCONTREI AQUI!",
                            "target_bbox": [53, 13940, 800, 14260],
                            "bubble_mask_bbox": [53, 13940, 800, 14260],
                            "balloon_bbox": [173, 13980, 733, 14220],
                            "bubble_mask_source": "real_bubble_mask",
                            "safe_text_box": [122, 14021, 656, 14179],
                            "_debug_safe_text_box": [122, 14021, 656, 14179],
                        }
                    ]
                }
            ]
        }

        audit = main._repair_project_real_bubble_body_safe_areas(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["safe_area_repaired_count"], 1)
        self.assertEqual(layer["layout_safe_reason"], "real_bubble_body_bbox")
        self.assertEqual(layer["layout_safe_bbox"], [229, 14016, 677, 14184])
        self.assertEqual(layer["safe_text_box"], [229, 14016, 677, 14184])
        self.assertEqual(layer["render_bbox"], [229, 14016, 677, 14184])
        self.assertTrue(layer["_render_bbox_from_repaired_safe_text_box"])
        self.assertIn("safe_text_box_recomputed", layer.get("qa_flags") or [])

    def test_repair_project_real_bubble_body_safe_area_replaces_safe_box_outside_real_inner_mask(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_003",
                            "translated": "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?",
                            "target_bbox": [29, 7109, 642, 7799],
                            "source_bbox": [325, 7647, 549, 7764],
                            "bbox": [344, 7702, 540, 7761],
                            "text_pixel_bbox": [344, 7702, 540, 7761],
                            "balloon_bbox": [276, 7612, 598, 7799],
                            "bubble_mask_bbox": [276, 7612, 598, 7799],
                            "bubble_inner_bbox": [337, 7659, 537, 7752],
                            "bubble_mask_source": "real_bubble_mask",
                            "safe_text_box": [98, 7229, 574, 7674],
                            "render_bbox": [190, 7428, 480, 7474],
                        }
                    ]
                }
            ]
        }

        audit = main._repair_project_real_bubble_body_safe_areas(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["safe_area_repaired_count"], 1)
        self.assertEqual(layer["layout_safe_reason"], "real_bubble_inner_bbox")
        self.assertEqual(layer["safe_text_box"], [357, 7675, 517, 7736])
        self.assertEqual(layer["render_bbox"], [357, 7675, 517, 7736])
        self.assertTrue(layer["_render_bbox_from_repaired_safe_text_box"])

    def test_repair_project_real_bubble_body_safe_area_does_not_trust_image_fallback_inner_mask(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_003",
                            "translated": "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?",
                            "target_bbox": [29, 7109, 642, 7799],
                            "source_bbox": [325, 7647, 549, 7764],
                            "bbox": [344, 7702, 540, 7761],
                            "text_pixel_bbox": [344, 7702, 540, 7761],
                            "balloon_bbox": [276, 7612, 598, 7799],
                            "bubble_mask_bbox": [276, 7612, 598, 7799],
                            "bubble_inner_bbox": [337, 7659, 537, 7752],
                            "bubble_mask_source": "image_white_bubble_mask",
                            "safe_text_box": [98, 7229, 574, 7674],
                            "render_bbox": [190, 7428, 480, 7474],
                        }
                    ]
                }
            ]
        }

        audit = main._repair_project_real_bubble_body_safe_areas(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["safe_area_repaired_count"], 0)
        self.assertEqual(layer["safe_text_box"], [98, 7229, 574, 7674])
        self.assertEqual(layer["render_bbox"], [190, 7428, 480, 7474])
        self.assertNotIn("_real_bubble_body_safe_area_repaired", layer)

    def test_repair_project_real_bubble_body_safe_area_does_not_trust_image_fallback_body_mask(self) -> None:
        project = {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "translated": "HYUNG-NIM! ENCONTREI AQUI!",
                            "target_bbox": [53, 13940, 800, 14260],
                            "bubble_mask_bbox": [53, 13940, 800, 14260],
                            "balloon_bbox": [173, 13980, 733, 14220],
                            "bubble_mask_source": "image_contour_bubble_mask",
                            "safe_text_box": [122, 14021, 656, 14179],
                            "_debug_safe_text_box": [122, 14021, 656, 14179],
                            "render_bbox": [122, 14021, 656, 14179],
                        }
                    ]
                }
            ]
        }

        audit = main._repair_project_real_bubble_body_safe_areas(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["safe_area_repaired_count"], 0)
        self.assertEqual(layer["safe_text_box"], [122, 14021, 656, 14179])
        self.assertEqual(layer["render_bbox"], [122, 14021, 656, 14179])
        self.assertNotIn("layout_safe_reason", layer)
        self.assertNotIn("_real_bubble_body_safe_area_repaired", layer)

    def test_final_renderer_keeps_repaired_real_bubble_geometry_despite_stale_flags(self) -> None:
        layer = {
            "id": "ocr_001",
            "translated": "HYUNG-NIM! ENCONTREI AQUI!",
            "target_bbox": [53, 13940, 800, 14260],
            "safe_text_box": [229, 14016, 677, 14184],
            "_debug_safe_text_box": [229, 14016, 677, 14184],
            "render_bbox": [229, 14016, 677, 14184],
            "layout_safe_reason": "real_bubble_body_bbox",
            "_render_bbox_from_repaired_safe_text_box": True,
            "_real_bubble_body_safe_area_repaired": True,
            "qa_flags": ["tiny_bubble_inner_bbox_rejected", "safe_text_box_recomputed"],
        }

        kept = main._drop_stale_final_render_geometry(dict(layer))

        self.assertEqual(kept["safe_text_box"], [229, 14016, 677, 14184])
        self.assertEqual(kept["render_bbox"], [229, 14016, 677, 14184])

    def test_sync_page_legacy_aliases_preserves_final_render_geometry_in_textos(self) -> None:
        page = {
            "numero": 1,
            "image_layers": {
                "base": {"path": "originals/001.jpg"},
                "rendered": {"path": "translated/001.jpg"},
            },
            "text_layers": [
                {
                    "id": "ocr_001",
                    "band_id": "page_002_band_007",
                    "original": "Quest Completion Criteria: Establish a Level 1 underworld.",
                    "translated": "Critérios de conclusão da missão: estabeleça um submundo de nível 1.",
                    "bbox": [82, 5389, 367, 5495],
                    "source_bbox": [82, 5389, 367, 5495],
                    "text_pixel_bbox": [82, 5389, 367, 5495],
                    "target_bbox": [20, 5348, 429, 5526],
                    "safe_text_box": [76, 5374, 373, 5487],
                    "_debug_safe_text_box": [76, 5374, 373, 5487],
                    "render_bbox": [82, 5381, 367, 5480],
                    "_debug_render_bbox": [82, 5381, 367, 5480],
                    "bubble_mask_source": "image_dark_bubble_mask",
                    "qa_flags": ["safe_text_box_recomputed"],
                }
            ],
        }

        main._sync_page_legacy_aliases(page)

        layer = page["text_layers"][0]
        legacy = page["textos"][0]
        self.assertEqual(layer["safe_text_box"], [76, 5374, 373, 5487])
        self.assertEqual(layer["render_bbox"], [82, 5381, 367, 5480])
        self.assertEqual(legacy["target_bbox"], [20, 5348, 429, 5526])
        self.assertEqual(legacy["safe_text_box"], [76, 5374, 373, 5487])
        self.assertEqual(legacy["render_bbox"], [82, 5381, 367, 5480])

    def test_final_page_space_renderer_drops_malformed_full_page_mixed_geometry(self) -> None:
        layers = [
            {
                "id": "valid_left",
                "band_id": "page_002_band_023",
                "translated": "Você era leal aos outros.",
                "bbox": [132, 18146, 557, 18355],
                "source_bbox": [132, 18146, 557, 18355],
                "text_pixel_bbox": [132, 18146, 557, 18355],
                "target_bbox": [63, 18030, 706, 18461],
                "safe_text_box": [224, 18138, 544, 18353],
                "render_bbox": [227, 18138, 542, 18317],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["safe_text_box_recomputed"],
            },
            {
                "id": "bad_mixed",
                "band_id": "page_002_band_023",
                "translated": "Texto duplicado misturado.",
                "bbox": [132, 18146, 649, 18392],
                "source_bbox": [132, 18146, 649, 18392],
                "text_pixel_bbox": [132, 18146, 649, 18392],
                "target_bbox": [63, 0, 740, 18397],
                "safe_text_box": [200, 94, 647, 18391],
                "render_bbox": [253, 116, 645, 18381],
                "qa_flags": [
                    "same_balloon_fragment_merged",
                    "debug_derived_bubble_mask_rejected",
                    "connected_lobe_boxes_missing_source_anchor_fallback",
                ],
            },
        ]

        normalized = main._final_page_space_text_layers_for_renderer(layers, page_number=1)

        self.assertEqual([layer["id"] for layer in normalized], ["valid_left"])

    def test_final_page_space_renderer_drops_local_duplicate_when_band_has_page_space_peer(self) -> None:
        layers = [
            {
                "id": "valid_page_space",
                "band_id": "page_002_band_023",
                "translated": "Você era o rei de ser simples.",
                "bbox": [476, 18283, 650, 18391],
                "source_bbox": [476, 18283, 650, 18391],
                "text_pixel_bbox": [476, 18283, 650, 18391],
                "target_bbox": [386, 18030, 740, 18489],
                "safe_text_box": [478, 18140, 647, 18379],
                "render_bbox": [480, 18195, 645, 18379],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["safe_text_box_recomputed"],
            },
            {
                "id": "stale_local_fragment",
                "band_id": "page_002_band_023",
                "translated": "Fragmento local duplicado.",
                "bbox": [476, 253, 649, 362],
                "source_bbox": [476, 253, 649, 362],
                "text_pixel_bbox": [476, 253, 649, 362],
                "target_bbox": [462, 247, 665, 363],
                "safe_text_box": [488, 272, 636, 333],
                "render_bbox": [503, 280, 623, 330],
                "qa_flags": [
                    "same_balloon_fragment_merged",
                    "connected_lobe_boxes_missing_source_anchor_fallback",
                ],
            },
        ]

        normalized = main._final_page_space_text_layers_for_renderer(layers, page_number=1)

        self.assertEqual([layer["id"] for layer in normalized], ["valid_page_space"])

    def test_final_renderer_anchors_repaired_real_bubble_to_safe_box(self) -> None:
        layers = [
            {
                "id": "ocr_001",
                "band_id": "page_002_band_016",
                "text": "HYUNGNIM! I FOUND IT OVER HERE!",
                "translated": "HYUNG-NIM! ENCONTREI AQUI!",
                "coordinate_space": "page",
                "source_coordinate_space": "page",
                "bbox": [350, 14052, 557, 14170],
                "layout_bbox": [350, 14052, 557, 14170],
                "target_bbox": [53, 13940, 800, 14260],
                "safe_text_box": [229, 14016, 677, 14184],
                "_debug_safe_text_box": [229, 14016, 677, 14184],
                "render_bbox": [229, 14016, 677, 14184],
                "layout_safe_reason": "real_bubble_body_bbox",
                "_render_bbox_from_repaired_safe_text_box": True,
                "_real_bubble_body_safe_area_repaired": True,
                "qa_flags": ["tiny_bubble_inner_bbox_rejected", "safe_text_box_recomputed"],
            }
        ]

        normalized = main._final_page_space_text_layers_for_renderer(layers, page_number=2)[0]

        self.assertEqual(normalized["bbox"], [229, 14016, 677, 14184])
        self.assertEqual(normalized["layout_bbox"], [229, 14016, 677, 14184])
        self.assertEqual(normalized["target_bbox"], [229, 14016, 677, 14184])
        self.assertTrue(normalized["_final_render_anchor_from_repaired_safe_text_box"])

    def test_debug_mask_bbox_repair_rejects_image_fallback_overbroad_cross_page_balloon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_019"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 328), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (242, 31, 580, 316))
            save_mask("05_balloon_inner_mask.png", (245, 33, 578, 304))
            save_mask("09_final_inpaint_mask.png", (292, 104, 532, 238))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_002_band_019",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": False,
                        "used_image_bubble_mask": True,
                        "used_derived_bubble_mask": False,
                        "used_balloon_bbox_fallback": False,
                        "bubble_mask_source": "image_white_bubble_mask",
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 2,
                        "width": 800,
                        "height": 16383,
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "band_id": "page_002_band_019",
                                "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                                "target_bbox": [298, 16276, 525, 16383],
                                "source_bbox": [298, 16276, 533, 16383],
                                "bbox": [298, 16287, 525, 16374],
                                "layout_bbox": [298, 16287, 525, 16374],
                                "text_pixel_bbox": [298, 16287, 525, 16374],
                                "balloon_bbox": [0, 16180, 800, 16383],
                                "bubble_mask_bbox": [0, 16180, 800, 16383],
                                "safe_text_box": [318, 16293, 504, 16383],
                                "render_bbox": [321, 16309, 503, 16376],
                                "qa_flags": ["rejected_derived_bubble_mask", "tiny_bubble_inner_bbox_rejected"],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["balloon_bbox"], [0, 16180, 800, 16383])
        self.assertEqual(layer["bubble_mask_bbox"], [0, 16180, 800, 16383])
        self.assertEqual(layer["safe_text_box"], [318, 16293, 504, 16383])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "untrusted_fallback_bubble_mask")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_final_page_space_normalization_shifts_local_render_geometry_from_page_source_ref(self) -> None:
        layer = {
            "id": "ocr_001",
            "band_id": "page_002_band_019",
            "coordinate_space": "page",
            "source_coordinate_space": "page",
            "source_bbox": [298, 16276, 533, 16383],
            "bbox": [298, 16287, 525, 16374],
            "layout_bbox": [298, 16287, 525, 16374],
            "text_pixel_bbox": [298, 16287, 525, 16374],
            "target_bbox": [298, 107, 525, 241],
            "safe_text_box": [318, 124, 504, 224],
            "_debug_safe_text_box": [318, 124, 504, 224],
            "render_bbox": [321, 140, 503, 207],
            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
            "qa_flags": ["same_balloon_fragment_merged"],
        }

        normalized = main._mark_final_layer_as_page_space(layer)

        self.assertEqual(normalized["target_bbox"], [298, 16276, 525, 16410])
        self.assertEqual(normalized["safe_text_box"], [318, 16293, 504, 16393])
        self.assertEqual(normalized["_debug_safe_text_box"], [318, 16293, 504, 16393])
        self.assertEqual(normalized["render_bbox"], [321, 16309, 503, 16376])
        self.assertEqual(normalized["source_bbox"], [298, 16276, 533, 16383])

        preserved = main._drop_stale_final_render_geometry(normalized)

        self.assertEqual(preserved["safe_text_box"], [318, 16293, 504, 16393])
        self.assertEqual(preserved["render_bbox"], [321, 16309, 503, 16376])

    def test_debug_mask_bbox_repair_rejects_high_outside_balloon_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_006_band_106"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 380), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (225, 175, 278, 205))
            save_mask("05_balloon_inner_mask.png", (227, 177, 276, 203))
            save_mask("09_final_inpaint_mask.png", (176, 173, 372, 215))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_006_band_106",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": True,
                        "used_image_bubble_mask": False,
                        "used_derived_bubble_mask": False,
                        "bubble_mask_source": "real_bubble_mask",
                        "synthetic_tight_balloon_reference": True,
                        "outside_balloon_pixels": 3679,
                        "outside_balloon_ratio": 0.748677,
                        "gates": {"mask_outside_balloon": True, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 6,
                        "width": 800,
                        "height": 15000,
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "band_id": "page_006_band_106",
                                "text_pixel_bbox": [178, 14378, 371, 14413],
                                "bbox": [178, 14378, 371, 14413],
                                "source_bbox": [178, 14378, 371, 14413],
                                "balloon_bbox": [0, 14280, 800, 14540],
                                "bubble_mask_bbox": [0, 14280, 800, 14540],
                                "safe_text_box": [178, 14378, 371, 14413],
                                "render_bbox": [178, 14378, 371, 14413],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["safe_text_box"], [178, 14378, 371, 14413])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "mask_outside_balloon")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_debug_mask_bbox_repair_rejects_image_source_claiming_real_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_008"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 360), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (40, 40, 760, 320))
            save_mask("05_balloon_inner_mask.png", (60, 60, 740, 300))
            save_mask("09_final_inpaint_mask.png", (290, 140, 510, 220))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_002_band_008",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": True,
                        "used_image_bubble_mask": False,
                        "used_derived_bubble_mask": False,
                        "bubble_mask_source": "image_white_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 2,
                        "width": 800,
                        "height": 12000,
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "band_id": "page_002_band_008",
                                "text_pixel_bbox": [290, 7140, 510, 7220],
                                "bbox": [290, 7140, 510, 7220],
                                "source_bbox": [290, 7140, 510, 7220],
                                "balloon_bbox": [40, 7040, 760, 7320],
                                "bubble_mask_bbox": [40, 7040, 760, 7320],
                                "safe_text_box": [290, 7140, 510, 7220],
                                "render_bbox": [290, 7140, 510, 7220],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "image_mask_misreported_as_real")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_debug_mask_bbox_repair_rejects_image_source_even_when_image_flag_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_008"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 360), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (40, 40, 760, 320))
            save_mask("05_balloon_inner_mask.png", (60, 60, 740, 300))
            save_mask("09_final_inpaint_mask.png", (290, 140, 510, 220))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_002_band_008",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": True,
                        "used_image_bubble_mask": True,
                        "used_derived_bubble_mask": False,
                        "bubble_mask_source": "image_white_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 2,
                        "width": 800,
                        "height": 12000,
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "band_id": "page_002_band_008",
                                "text_pixel_bbox": [290, 7140, 510, 7220],
                                "bbox": [290, 7140, 510, 7220],
                                "source_bbox": [290, 7140, 510, 7220],
                                "balloon_bbox": [40, 7040, 760, 7320],
                                "bubble_mask_bbox": [40, 7040, 760, 7320],
                                "safe_text_box": [290, 7140, 510, 7220],
                                "render_bbox": [290, 7140, 510, 7220],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "image_mask_misreported_as_real")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_debug_mask_bbox_repair_rejects_untrusted_derived_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_006_band_106"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 380), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (225, 175, 278, 205))
            save_mask("05_balloon_inner_mask.png", (227, 177, 276, 203))
            save_mask("09_final_inpaint_mask.png", (176, 173, 372, 215))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_006_band_106",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": False,
                        "used_derived_bubble_mask": True,
                        "bubble_mask_source": "derived_white_crop",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 6,
                        "width": 800,
                        "height": 15000,
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "band_id": "page_006_band_106",
                                "text_pixel_bbox": [178, 14378, 371, 14413],
                                "bbox": [178, 14378, 371, 14413],
                                "source_bbox": [178, 14378, 371, 14413],
                                "balloon_bbox": [0, 14280, 800, 14540],
                                "bubble_mask_bbox": [0, 14280, 800, 14540],
                                "safe_text_box": [178, 14378, 371, 14413],
                                "render_bbox": [178, 14378, 371, 14413],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["safe_text_box"], [178, 14378, 371, 14413])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "untrusted_derived_bubble_mask")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_debug_mask_bbox_repair_rejects_multi_text_band_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_003_band_035"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 900), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (133, 30, 598, 760))
            save_mask("05_balloon_inner_mask.png", (135, 32, 596, 752))
            save_mask("09_final_inpaint_mask.png", (344, 682, 540, 741))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_003_band_035",
                        "text_ids": ["ocr_001", "ocr_003"],
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": True,
                        "used_image_bubble_mask": False,
                        "used_derived_bubble_mask": False,
                        "bubble_mask_source": "real_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 3,
                        "width": 800,
                        "height": 12000,
                        "text_layers": [
                            {
                                "id": "ocr_003",
                                "band_id": "page_003_band_035",
                                "text_pixel_bbox": [344, 7702, 540, 7761],
                                "bbox": [344, 7702, 540, 7761],
                                "source_bbox": [344, 7702, 540, 7761],
                                "balloon_bbox": [0, 7000, 800, 8000],
                                "bubble_mask_bbox": [0, 7000, 800, 8000],
                                "safe_text_box": [344, 7702, 540, 7761],
                                "render_bbox": [344, 7702, 540, 7761],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["safe_text_box"], [344, 7702, 540, 7761])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "multi_text_debug_mask_not_per_layer")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_debug_mask_bbox_repair_uses_per_text_mask_when_band_mask_is_multi_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_003_band_035"
            per_text_dir = band_dir / "per_text" / "ocr_003"
            band_dir.mkdir(parents=True)
            per_text_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(directory: Path, name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (160, 100), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(directory / name)

            save_mask(band_dir, "04_balloon_mask.png", (10, 10, 150, 90))
            save_mask(band_dir, "05_balloon_inner_mask.png", (12, 12, 148, 88))
            save_mask(band_dir, "09_final_inpaint_mask.png", (20, 20, 132, 58))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_003_band_035",
                        "text_ids": ["ocr_001", "ocr_003"],
                        "used_balloon_clip": True,
                        "used_image_bubble_mask": True,
                        "bubble_mask_source": "image_white_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            save_mask(per_text_dir, "04_balloon_mask.png", (96, 40, 148, 70))
            save_mask(per_text_dir, "05_balloon_inner_mask.png", (100, 44, 144, 66))
            save_mask(per_text_dir, "09_final_inpaint_mask.png", (108, 50, 132, 58))
            (per_text_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_003_band_035",
                        "text_id": "ocr_003",
                        "text_ids": ["ocr_003"],
                        "used_balloon_clip": True,
                        "used_image_bubble_mask": True,
                        "bubble_mask_source": "image_white_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 3,
                        "width": 160,
                        "height": 8000,
                        "text_layers": [
                            {
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_035",
                                "band_id": "page_003_band_035",
                                "text_pixel_bbox": [108, 7050, 132, 7058],
                                "bbox": [108, 7050, 132, 7058],
                                "source_bbox": [108, 7050, 132, 7058],
                                "balloon_bbox": [0, 7000, 160, 7100],
                                "bubble_mask_bbox": [0, 7000, 160, 7100],
                                "safe_text_box": [108, 7050, 132, 7058],
                                "render_bbox": [108, 7050, 132, 7058],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 1)
        self.assertEqual(layer["bubble_mask_bbox"], [96, 7040, 149, 7071])
        self.assertEqual(layer["bubble_inner_bbox"], [100, 7044, 145, 7067])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_unclamped")
        self.assertTrue(layer["_debug_derived_bubble_bbox_repaired"])
        self.assertNotEqual(layer.get("_debug_derived_bubble_bbox_rejected"), "multi_text_debug_mask_not_per_layer")

    def test_debug_mask_bbox_repair_rejects_degenerate_restored_safe_text_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            band_dir = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_006_band_106"
            band_dir.mkdir(parents=True)
            from PIL import Image, ImageDraw

            def save_mask(name: str, bbox: tuple[int, int, int, int]) -> None:
                image = Image.new("L", (800, 380), 0)
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox, fill=255)
                image.save(band_dir / name)

            save_mask("04_balloon_mask.png", (225, 175, 278, 205))
            save_mask("05_balloon_inner_mask.png", (227, 177, 276, 203))
            save_mask("09_final_inpaint_mask.png", (176, 173, 372, 215))
            (band_dir / "mask_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_006_band_106",
                        "used_balloon_clip": True,
                        "used_real_bubble_mask": True,
                        "used_image_bubble_mask": False,
                        "used_derived_bubble_mask": False,
                        "bubble_mask_source": "real_bubble_mask",
                        "outside_balloon_pixels": 0,
                        "outside_balloon_ratio": 0.0,
                        "gates": {"mask_outside_balloon": False, "mask_outside_balloon_critical": False},
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 6,
                        "width": 800,
                        "height": 15000,
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "band_id": "page_006_band_106",
                                "text_pixel_bbox": [178, 14378, 371, 14413],
                                "bbox": [178, 14378, 371, 14413],
                                "source_bbox": [178, 14378, 371, 14413],
                                "balloon_bbox": [0, 14280, 800, 14540],
                                "bubble_mask_bbox": [0, 14280, 800, 14540],
                                "safe_text_box": [178, 14378, 371, 14413],
                                "render_bbox": [178, 14378, 371, 14413],
                            }
                        ],
                    }
                ],
            }

            audit = main._repair_project_bubble_bboxes_from_debug_masks(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["layers_repaired"], 0)
        self.assertEqual(layer["safe_text_box"], [178, 14378, 371, 14413])
        self.assertEqual(layer["layout_safe_reason"], "debug_derived_bubble_mask_rejected")
        self.assertEqual(layer["_debug_derived_bubble_bbox_rejected"], "safe_text_box_degenerate")
        self.assertIn("debug_derived_bubble_mask_rejected", layer.get("qa_flags") or [])
        self.assertNotIn("_debug_derived_bubble_bbox_repaired", layer)

    def test_hydrate_project_render_metadata_keeps_two_balloons_in_same_band_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            raw_rows = [
                {
                    "trace_id": "ocr_002@page_003_band_039",
                    "text_id": "ocr_002",
                    "band_id": "page_003_band_039",
                    "page_id": "page_003",
                    "text": "I'M SORRY, IN-HONG...",
                    "translated": "ME DESCULPE, IN-HONG...",
                    "target_bbox": [74, 134, 287, 252],
                    "safe_text_box": [100, 158, 260, 228],
                    "render_bbox": [101, 172, 259, 214],
                    "bbox": [106, 156, 255, 230],
                    "layout_bbox": [115, 174, 248, 227],
                    "text_pixel_bbox": [115, 174, 248, 227],
                    "fit_status": "ok",
                },
                {
                    "trace_id": "ocr_003@page_003_band_039",
                    "text_id": "ocr_003",
                    "band_id": "page_003_band_039",
                    "page_id": "page_003",
                    "text": "MOM IS SORRY.",
                    "translated": "MAMÃE ESTÁ ARREPENDIDA.",
                    "target_bbox": [487, 186, 749, 253],
                    "safe_text_box": [510, 193, 727, 246],
                    "render_bbox": [547, 198, 688, 240],
                    "bbox": [527, 198, 709, 241],
                    "layout_bbox": [528, 213, 714, 234],
                    "text_pixel_bbox": [528, 213, 714, 234],
                    "fit_status": "ok",
                },
            ]
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "trace_id": "ocr_002@page_003_band_039",
                                "text_id": "ocr_002",
                                "band_id": "page_003_band_039",
                                "page_id": "page_003",
                                "bboxes": {"bbox": {"value": [74, 70, 287, 188]}},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "trace_id": "ocr_003@page_003_band_039",
                                "text_id": "ocr_003",
                                "band_id": "page_003_band_039",
                                "page_id": "page_003",
                                "bboxes": {"bbox": {"value": [487, 122, 749, 189]}},
                            },
                            ensure_ascii=False,
                        ),
                    ]
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_039",
                                "band_id": "page_003_band_039",
                                "page_id": "page_003",
                                "route_action": "translate_inpaint_render",
                                "text": "I'M SORRY, IN-HONG... MOM IS SORRY. 3 TI2]2H",
                                "translated": "ME DESCULPE, IN-HONG...",
                                "bbox": [74, 70, 287, 188],
                                "target_bbox": [106, 92, 781, 211],
                                "safe_text_box": [132, 116, 759, 204],
                                "render_bbox": [133, 130, 720, 198],
                                "text_pixel_bbox": [115, 110, 714, 9252],
                                "balloon_bbox": [74, 70, 287, 188],
                                "qa_flags": [],
                            },
                            {
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_039",
                                "band_id": "page_003_band_039",
                                "page_id": "page_003",
                                "route_action": "translate_inpaint_render",
                                "text": "MOM IS SORRY.",
                                "translated": "MAMÃE ESTÁ ARREPENDIDA.",
                                "bbox": [487, 122, 749, 189],
                                "text_pixel_bbox": [487, 122, 749, 189],
                                "balloon_bbox": [487, 122, 749, 189],
                                "qa_flags": ["missing_render_bbox"],
                            },
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        left, right = project["paginas"][0]["text_layers"]
        self.assertEqual(audit["hydrated_layers"], 2)
        self.assertEqual(left["translated"], "ME DESCULPE, IN-HONG...")
        self.assertEqual(left["target_bbox"], [74, 134, 287, 252])
        self.assertEqual(left["safe_text_box"], [100, 158, 260, 228])
        self.assertEqual(left["render_bbox"], [101, 172, 259, 214])
        self.assertEqual(right["translated"], "MAMÃE ESTÁ ARREPENDIDA.")
        self.assertEqual(right["target_bbox"], [487, 122, 749, 189])
        self.assertEqual(right["safe_text_box"], [510, 129, 727, 182])
        self.assertEqual(right["render_bbox"], [547, 134, 688, 176])
        self.assertNotEqual(left["safe_text_box"], [100, 158, 727, 246])
        self.assertLess(left["safe_text_box"][2], right["safe_text_box"][0])
        self.assertNotEqual(left.get("render_policy"), "merged_into_primary")
        self.assertNotEqual(right.get("render_policy"), "merged_into_primary")
        self.assertIsNot(left.get("visible"), False)
        self.assertIsNot(right.get("visible"), False)
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_restores_missing_same_band_sibling_from_raw_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            raw_rows = [
                {
                    "trace_id": "ocr_002@page_003_band_039",
                    "text_id": "ocr_002",
                    "band_id": "page_003_band_039",
                    "page_id": "page_003",
                    "text": "I'M SORRY, IN-HONG...",
                    "translated": "ME DESCULPE, IN-HONG...",
                    "target_bbox": [74, 134, 287, 252],
                    "safe_text_box": [100, 158, 260, 228],
                    "render_bbox": [101, 172, 259, 214],
                    "bbox": [106, 156, 255, 230],
                    "layout_bbox": [115, 174, 248, 227],
                    "text_pixel_bbox": [115, 174, 248, 227],
                    "fit_status": "ok",
                },
                {
                    "trace_id": "ocr_003@page_003_band_039",
                    "text_id": "ocr_003",
                    "band_id": "page_003_band_039",
                    "page_id": "page_003",
                    "text": "MOM IS SORRY.",
                    "translated": "MAMÃE ESTÁ ARREPENDIDA.",
                    "target_bbox": [487, 186, 749, 253],
                    "safe_text_box": [510, 193, 727, 246],
                    "render_bbox": [547, 198, 688, 240],
                    "bbox": [527, 198, 709, 241],
                    "layout_bbox": [528, 213, 714, 234],
                    "text_pixel_bbox": [528, 213, 714, 234],
                    "fit_status": "ok",
                },
            ]
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_039",
                                "band_id": "page_003_band_039",
                                "page_id": "page_003",
                                "route_action": "translate_inpaint_render",
                                "text": "I'M SORRY, IN-HONG... MOM IS SORRY. 3 TI2]2H",
                                "translated": "ME DESCULPE, IN-HONG...",
                                "bbox": [115, 110, 248, 163],
                                "source_bbox": [106, 92, 709, 9248],
                                "layout_bbox": [115, 110, 248, 163],
                                "balloon_bbox": [74, 70, 287, 188],
                                "text_pixel_bbox": [115, 110, 714, 9252],
                                "target_bbox": [106, 92, 781, 211],
                                "safe_text_box": [132, 116, 759, 204],
                                "render_bbox": [133, 130, 720, 198],
                                "qa_flags": ["ocr_gibberish"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layers = project["paginas"][0]["text_layers"]
        self.assertEqual(audit["restored_missing_candidate_layers"], 1)
        self.assertEqual(audit["hydrated_layers"], 2)
        self.assertEqual([layer["text_id"] for layer in layers], ["ocr_002", "ocr_003"])
        self.assertEqual(layers[0]["target_bbox"], [74, 70, 287, 188])
        self.assertEqual(layers[0]["safe_text_box"], [100, 94, 260, 164])
        self.assertEqual(layers[0]["text_pixel_bbox"], [115, 110, 248, 163])
        self.assertEqual(layers[0]["source_bbox"], [115, 110, 248, 163])
        self.assertEqual(layers[1]["translated"], "MAMÃE ESTÁ ARREPENDIDA.")
        self.assertEqual(layers[1]["target_bbox"], [487, 122, 749, 189])
        self.assertEqual(layers[1]["safe_text_box"], [510, 129, 727, 182])
        self.assertEqual(layers[1]["render_bbox"], [547, 134, 688, 176])
        self.assertEqual(layers[1]["text_pixel_bbox"], [528, 149, 714, 170])
        self.assertEqual(layers[1].get("_same_band_restore_coordinate_offset"), [0, -64])
        self.assertTrue(layers[1].get("_restored_from_render_plan_candidate"))
        self.assertEqual(contract["missing_render_bbox_count"], 0)
        final_layers = main._final_page_space_text_layers_for_renderer(layers, page_number=3)
        self.assertEqual(len(final_layers), 2)
        self.assertEqual(final_layers[0]["text_id"], "ocr_002")
        self.assertEqual(final_layers[1]["text_id"], "ocr_003")
        self.assertEqual(final_layers[0]["safe_text_box"], layers[0]["safe_text_box"])
        self.assertEqual(final_layers[1]["safe_text_box"], [510, 129, 727, 182])
        self.assertEqual(final_layers[1]["render_bbox"], [547, 134, 688, 176])
        from typesetter.renderer import build_render_blocks

        render_blocks = [
            block
            for block in build_render_blocks(final_layers)
            if block.get("band_id") == "page_003_band_039"
        ]
        self.assertEqual([block["translated"] for block in render_blocks], ["ME DESCULPE, IN-HONG...", layers[1]["translated"]])

    def test_main_final_page_space_rerender_uses_page_coordinate_space(self) -> None:
        source = Path(main.__file__).read_text(encoding="utf-8")

        self.assertIn('{"texts": page_texts, "_coordinate_space": "page"}', source)
        self.assertIn('{"texts": _visible_render_texts(trans_texts), "_coordinate_space": "page"}', source)
        self.assertNotIn('"_coordinate_space": "main_final_page_space_typeset"', source)

    def test_main_final_page_space_typeset_is_opt_in(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            main.os.environ.pop("TRADUZAI_MAIN_FINAL_PAGE_SPACE_TYPESET", None)
            self.assertFalse(main._main_final_page_space_typeset_enabled())
        with patch.dict("os.environ", {"TRADUZAI_MAIN_FINAL_PAGE_SPACE_TYPESET": "1"}):
            self.assertTrue(main._main_final_page_space_typeset_enabled())

    def test_visible_render_texts_keeps_translated_layer_hidden_by_fit_failure(self) -> None:
        layers = [
            {
                "id": "ocr_001",
                "translated": "BEM... NAO E COMO SE ISSO NAO FOSSE AGRADAVEL...",
                "original": "Well... it's not like this isn't delightful...",
                "visible": False,
                "fit_status": "below_minimum_legible",
                "qa_flags": ["missing_render_bbox"],
            },
            {
                "id": "ocr_002",
                "translated": "BEM... NAO E COMO SE ISSO NAO FOSSE AGRADAVEL... DCCIGHTPOC..",
                "visible": False,
                "render_policy": "merged_into_primary",
            },
        ]

        visible = main._visible_render_texts(layers)

        self.assertEqual([layer["id"] for layer in visible], ["ocr_001"])

    def test_hydrate_project_render_metadata_uses_safe_box_when_raw_render_bbox_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_023",
                        "text_id": "ocr_001",
                        "band_id": "page_003_band_023",
                        "page_id": "page_003",
                        "translated": "NÃO APERTE MINHA MÃE!",
                        "target_bbox": [0, 96, 555, 755],
                        "render_bbox": None,
                        "safe_text_box": [82, 209, 474, 642],
                        "fit_status": "ok",
                        "qa_flags": ["unsafe_derived_art_mask_review"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_023",
                        "text_id": "ocr_001",
                        "band_id": "page_003_band_023",
                        "page_id": "page_003",
                        "bboxes": {"bbox": {"value": [0, 5320, 555, 5979]}},
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
                                "trace_id": "ocr_001@page_003_band_023",
                                "band_id": "page_003_band_023",
                                "page_id": "page_003",
                                "route_action": "review_required",
                                "translated": "NÃO APERTE MINHA MÃE!",
                                "bbox": [0, 5320, 555, 5979],
                                "balloon_bbox": [0, 5320, 555, 5979],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["safe_text_box"], [82, 5433, 474, 5866])
        self.assertEqual(layer["render_bbox"], [82, 5433, 474, 5866])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_can_match_merged_band_candidate_by_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "source_trace_ids": ["ocr_001@page_003_band_028"],
                        "source_text_ids": ["ocr_001"],
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "translated": "suas\nA imagem virtual é semelhante a Langit.",
                        "target_bbox": [242, 390, 642, 566],
                        "render_bbox": [330, 451, 553, 504],
                        "safe_text_box": [303, 425, 581, 505],
                        "fit_status": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "bboxes": {"bbox": {"value": [242, 4917, 642, 5093]}},
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
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_028",
                                "band_id": "page_003_band_028",
                                "page_id": "page_003",
                                "route_action": "translate_inpaint_render",
                                "translated": "A imagem virtual é semelhante a Langit.",
                                "bbox": [303, 4957, 581, 5067],
                                "text_pixel_bbox": [303, 4957, 581, 5067],
                                "balloon_bbox": [242, 4951, 642, 5093],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["target_bbox"], [242, 4951, 642, 5127])
        self.assertEqual(layer["render_bbox"], [330, 5012, 553, 5065])
        self.assertEqual(layer["safe_text_box"], [303, 4986, 581, 5066])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_keeps_page_space_candidate_when_layout_offset_is_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "source_trace_ids": ["ocr_001@page_003_band_028"],
                        "source_text_ids": ["ocr_001"],
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "translated": "suas\nA imagem virtual é semelhante a Langit.",
                        "target_bbox": [242, 4917, 642, 5093],
                        "render_bbox": [330, 4919, 553, 5038],
                        "safe_text_box": [303, 4917, 581, 5067],
                        "fit_status": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "bboxes": {"bbox": {"value": [191, 4640, 448, 4793]}},
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
                                "id": "ocr_003",
                                "text_id": "ocr_003",
                                "trace_id": "ocr_003@page_003_band_028",
                                "band_id": "page_003_band_028",
                                "page_id": "page_003",
                                "route_action": "translate_inpaint_render",
                                "translated": "A imagem virtual é semelhante a Langit.",
                                "bbox": [303, 4957, 581, 5067],
                                "text_pixel_bbox": [303, 4957, 581, 5067],
                                "balloon_bbox": [242, 4951, 642, 5093],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["target_bbox"], [242, 4917, 642, 5093])
        self.assertEqual(layer["render_bbox"], [330, 4919, 553, 5038])
        self.assertEqual(layer["safe_text_box"], [303, 4917, 581, 5067])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_does_not_cross_match_bare_text_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_candidates.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "trace_id": "ocr_002@page_003_band_027",
                                "text_id": "ocr_002",
                                "band_id": "page_003_band_027",
                                "page_id": "page_003",
                                "translated": "FOTO ESTRELA",
                                "target_bbox": [342, 224, 626, 297],
                                "render_bbox": [354, 247, 624, 274],
                                "safe_text_box": [348, 227, 625, 294],
                            }
                        ),
                        json.dumps(
                            {
                                "trace_id": "ocr_002@page_019_band_191",
                                "text_id": "ocr_002",
                                "band_id": "page_019_band_191",
                                "page_id": "page_019",
                                "translated": "POSSO FAZER O MEU",
                                "target_bbox": [169, 137, 1521, 927],
                                "render_bbox": [599, 532, 1015, 562],
                                "safe_text_box": [452, 259, 1162, 836],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_002@page_003_band_027",
                        "text_id": "ocr_002",
                        "band_id": "page_003_band_027",
                        "page_id": "page_003",
                        "bboxes": {"bbox": {"value": [342, 2247, 626, 2320]}},
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "trace_id": "ocr_002@page_019_band_191",
                        "text_id": "ocr_002",
                        "band_id": "page_019_band_191",
                        "page_id": "page_019",
                        "bboxes": {"bbox": {"value": [169, 1654, 1521, 2444]}},
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_027",
                                "band_id": "page_003_band_027",
                                "page_id": "page_003",
                                "translated": "FOTO ESTRELA",
                                "bbox": [341, 2247, 621, 2323],
                                "text_pixel_bbox": [348, 2251, 625, 2318],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["target_bbox"], [342, 2248, 626, 2321])
        self.assertEqual(layer["render_bbox"], [354, 2271, 624, 2298])
        self.assertNotEqual(layer["render_bbox"], [599, 2049, 1015, 2079])

    def test_hydrate_project_render_metadata_repairs_existing_displaced_render_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_002@page_003_band_027",
                        "text_id": "ocr_002",
                        "band_id": "page_003_band_027",
                        "page_id": "page_003",
                        "translated": "FOTO ESTRELA",
                        "target_bbox": [342, 224, 626, 297],
                        "render_bbox": [354, 247, 624, 274],
                        "safe_text_box": [348, 227, 625, 294],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (debug_root / "05_layout_geometry" / "layout_blocks.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_002@page_003_band_027",
                        "text_id": "ocr_002",
                        "band_id": "page_003_band_027",
                        "page_id": "page_003",
                        "bboxes": {"bbox": {"value": [342, 2247, 626, 2320]}},
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_027",
                                "band_id": "page_003_band_027",
                                "page_id": "page_003",
                                "translated": "FOTO ESTRELA",
                                "bbox": [341, 2247, 621, 2323],
                                "text_pixel_bbox": [348, 2251, 625, 2318],
                                "safe_text_box": [452, 1776, 1162, 2353],
                                "render_bbox": [599, 2049, 1015, 2079],
                                "qa_flags": ["fast_fill_no_glyph_evidence"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [354, 2271, 624, 2298])
        self.assertEqual(layer["safe_text_box"], [348, 2251, 625, 2318])

    def test_hydrate_project_render_metadata_aggregates_merged_child_render_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        {
                            "trace_id": "ocr_003@page_003_band_031",
                            "text_id": "ocr_003",
                            "band_id": "page_003_band_031",
                            "page_id": "page_003",
                            "translated": "TÍTULO DE OUTRA BANDA",
                            "target_bbox": [200, 490, 1200, 650],
                            "safe_text_box": [240, 510, 1160, 630],
                            "render_bbox": [300, 540, 1100, 590],
                            "fit_status": "ok",
                        },
                        {
                            "trace_id": "ocr_001_uied_label_01@page_004_band_043",
                            "text_id": "ocr_001_uied_label_01",
                            "band_id": "page_004_band_043",
                            "page_id": "page_004",
                            "translated": "MAS EU VIM",
                            "target_bbox": [808, 107, 1138, 142],
                            "safe_text_box": [822, 107, 1125, 142],
                            "render_bbox": [886, 114, 1060, 134],
                            "fit_status": "ok",
                        },
                        {
                            "trace_id": "ocr_001_uied_label_02@page_004_band_043",
                            "text_id": "ocr_001_uied_label_02",
                            "band_id": "page_004_band_043",
                            "page_id": "page_004",
                            "translated": "ATÉ AQUI!",
                            "target_bbox": [755, 154, 1189, 197],
                            "safe_text_box": [772, 154, 1171, 197],
                            "render_bbox": [883, 157, 1060, 193],
                            "fit_status": "ok",
                        },
                        {
                            "trace_id": "ocr_001_uied_label_03@page_004_band_043",
                            "text_id": "ocr_001_uied_label_03",
                            "band_id": "page_004_band_043",
                            "page_id": "page_004",
                            "translated": "EU NÃO POSSO PARECER",
                            "target_bbox": [768, 209, 1174, 243],
                            "safe_text_box": [784, 209, 1157, 243],
                            "render_bbox": [805, 212, 1137, 239],
                            "fit_status": "ok",
                        },
                    ]
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
                                "id": "ocr_001_uied_label_02",
                                "text_id": "ocr_001_uied_label_02",
                                "trace_id": "ocr_001_uied_label_02@page_004_band_043",
                                "source_trace_ids": [
                                    "ocr_003@page_003_band_031",
                                    "ocr_001_uied_label_01@page_004_band_043",
                                    "ocr_001_uied_label_02@page_004_band_043",
                                    "ocr_001_uied_label_03@page_004_band_043",
                                ],
                                "source_text_ids": [
                                    "ocr_001_uied_label_01",
                                    "ocr_001_uied_label_02",
                                    "ocr_001_uied_label_03",
                                ],
                                "band_id": "page_004_band_043",
                                "page_id": "page_004",
                                "route_action": "translate_inpaint_render",
                                "translated": "MAS EU VIM ATÉ AQUI! EU NÃO POSSO PARECER",
                                "bbox": [755, 5961, 1189, 6151],
                                "source_bbox": [755, 5961, 1189, 6151],
                                "text_pixel_bbox": [755, 5961, 1189, 6151],
                                "layout_bbox": [755, 6008, 1189, 6051],
                                "balloon_bbox": [523, 5902, 1421, 6110],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [805, 5968, 1137, 6093])
        self.assertEqual(layer["safe_text_box"], [772, 5961, 1171, 6097])
        self.assertEqual(layer["fit_status"], "ok")
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_aggregates_same_layer_split_render_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        {
                            "trace_id": "ocr_002@page_002_band_031",
                            "text_id": "ocr_002",
                            "band_id": "page_002_band_031",
                            "page_id": "page_002",
                            "translated": "O QUE ACONTECEU?",
                            "target_bbox": [232, 112, 385, 179],
                            "safe_text_box": [254, 124, 364, 167],
                            "render_bbox": [256, 129, 361, 162],
                            "fit_status": "ok",
                        },
                        {
                            "trace_id": "ocr_002@page_002_band_031",
                            "text_id": "ocr_002",
                            "band_id": "page_002_band_031",
                            "page_id": "page_002",
                            "translated": "VOCE NAO FUGIU?",
                            "target_bbox": [392, 273, 543, 337],
                            "safe_text_box": [414, 285, 522, 325],
                            "render_bbox": [431, 288, 504, 322],
                            "fit_status": "ok",
                        },
                    ]
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_002_band_031",
                                "band_id": "page_002_band_031",
                                "page_id": "page_002",
                                "route_action": "translate_inpaint_render",
                                "translated": "O QUE ACONTECEU? VOCE NAO FUGIU?",
                                "bbox": [232, 4954, 546, 5198],
                                "source_bbox": [232, 4954, 546, 5198],
                                "text_pixel_bbox": [232, 4954, 546, 5198],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [256, 4971, 504, 5164])
        self.assertEqual(layer["safe_text_box"], [254, 4966, 522, 5167])
        self.assertEqual(layer["target_bbox"], [232, 4954, 543, 5179])
        self.assertEqual(layer["fit_status"], "ok")
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_uses_text_geometry_bbox_for_band_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "05_layout_geometry").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_002@page_025_band_258",
                        "text_id": "ocr_002",
                        "source_trace_ids": ["ocr_002@page_025_band_258", "ocr_003@page_025_band_258"],
                        "band_id": "page_025_band_258",
                        "page_id": "page_025",
                        "translated": "ELE PODE RECEBER UMA OFERTA DE PROJETO. TIRE UMA FOLGA",
                        "target_bbox": [563, 433, 1048, 618],
                        "safe_text_box": [604, 460, 1006, 591],
                        "render_bbox": [608, 477, 1004, 574],
                        "fit_status": "ok",
                        "qa_metrics": {
                            "bbox_overreach": {
                                "text_geometry_bbox": [563, 433, 1049, 619],
                                "broad_bbox_drives_mask": False,
                            }
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_025_band_258",
                                "source_trace_ids": ["ocr_002@page_025_band_258", "ocr_003@page_025_band_258"],
                                "band_id": "page_025_band_258",
                                "page_id": "page_025",
                                "route_action": "translate_inpaint_render",
                                "translated": "ELE PODE RECEBER UMA OFERTA DE PROJETO. TIRE UMA FOLGA",
                                "bbox": [357, 6719, 1059, 7220],
                                "source_bbox": [357, 6719, 1059, 7220],
                                "layout_bbox": [361, 6725, 1048, 7142],
                                "balloon_bbox": [357, 6802, 802, 7220],
                                "qa_metrics": {
                                    "bbox_overreach": {
                                        "text_geometry_bbox": [563, 6725, 1049, 6911],
                                        "broad_bbox_drives_mask": False,
                                    }
                                },
                                "qa_flags": ["missing_render_bbox", "mask_outside_balloon_critical"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [608, 6769, 1004, 6866])
        self.assertEqual(layer["safe_text_box"], [604, 6752, 1006, 6883])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])

    def test_hydrate_project_render_metadata_does_not_copy_band_local_balloon_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_004@page_002_band_002",
                        "text_id": "ocr_004",
                        "band_id": "page_002_band_002",
                        "page_id": "page_002",
                        "translated": "POR FAVOR, ESPERE MAIS ALGUNS DIAS!",
                        "target_bbox": [64, 2616, 372, 2782],
                        "safe_text_box": [141, 2645, 291, 2753],
                        "render_bbox": [160, 2670, 280, 2725],
                        "balloon_bbox": [64, 1075, 372, 1241],
                        "bubble_mask_bbox": [64, 2616, 372, 2782],
                        "fit_status": "ok",
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
                                "id": "ocr_004",
                                "text_id": "ocr_004",
                                "trace_id": "ocr_004@page_002_band_002",
                                "band_id": "page_002_band_002",
                                "page_id": "page_002",
                                "route_action": "translate_inpaint_render",
                                "translated": "POR FAVOR, ESPERE MAIS ALGUNS DIAS!",
                                "bbox": [114, 2647, 325, 2748],
                                "source_bbox": [114, 2647, 325, 2748],
                                "layout_bbox": [114, 2647, 325, 2748],
                                "balloon_bbox": [64, 2616, 372, 2782],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["balloon_bbox"], [64, 2616, 372, 2782])
        self.assertEqual(layer["bubble_mask_bbox"], [64, 2616, 372, 2782])
        self.assertEqual(layer["render_bbox"], [160, 2670, 280, 2725])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])

    def test_hydrate_project_render_metadata_trusts_explicit_group_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "source_text_ids": ["ocr_001", "ocr_002"],
                        "source_trace_ids": ["ocr_001@page_003_band_028", "ocr_002@page_003_band_028"],
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "translated": "TEXTO LONGO DO BLOCO\nNOTÍCIAS DIÁRIAS",
                        "render_bbox": [230, 2579, 1368, 2720],
                        "safe_text_box": [99, 2420, 1499, 2879],
                        "fit_status": "ok",
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
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_028",
                                "band_id": "page_003_band_028",
                                "translated": "NOTÍCIAS DIÁRIAS",
                                "bbox": [424, 2768, 748, 2832],
                                "text_pixel_bbox": [416, 2768, 750, 2835],
                                "qa_flags": ["missing_render_bbox"],
                            }
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [230, 2579, 1368, 2720])
        self.assertEqual(layer["safe_text_box"], [99, 2420, 1499, 2879])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])

    def test_hydrate_project_render_metadata_copies_group_sibling_page_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_003_band_028",
                        "text_id": "ocr_001",
                        "source_text_ids": ["ocr_001", "ocr_002"],
                        "source_trace_ids": ["ocr_001@page_003_band_028", "ocr_002@page_003_band_028"],
                        "band_id": "page_003_band_028",
                        "page_id": "page_003",
                        "translated": "TEXTO LONGO DO BLOCO\nNOTÍCIAS DIÁRIAS",
                        "render_bbox": [230, 218, 1368, 359],
                        "safe_text_box": [99, 59, 1499, 518],
                        "fit_status": "ok",
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
                                "trace_id": "ocr_001@page_003_band_028",
                                "band_id": "page_003_band_028",
                                "translated": "TEXTO LONGO DO BLOCO",
                                "bbox": [376, 2457, 1228, 2826],
                                "render_bbox": [230, 2579, 1368, 2720],
                                "safe_text_box": [99, 2420, 1499, 2879],
                                "fit_status": "ok",
                            },
                            {
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_003_band_028",
                                "band_id": "page_003_band_028",
                                "translated": "NOTÍCIAS DIÁRIAS",
                                "bbox": [424, 2768, 748, 2832],
                                "qa_flags": ["missing_render_bbox"],
                            },
                        ]
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        layer = project["paginas"][0]["text_layers"][1]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertEqual(layer["render_bbox"], [230, 2579, 1368, 2720])
        self.assertEqual(layer["safe_text_box"], [99, 2420, 1499, 2879])
        self.assertNotIn("missing_render_bbox", layer.get("qa_flags") or [])

    def test_hydrate_project_render_metadata_collapses_merged_source_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_002@page_002_band_019",
                        "text_id": "ocr_002",
                        "source_text_ids": ["ocr_002", "ocr_001"],
                        "source_trace_ids": [
                            "ocr_002@page_002_band_019",
                            "ocr_001@page_002_band_019",
                        ],
                        "band_id": "page_002_band_019",
                        "page_id": "page_002",
                        "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                        "target_bbox": [298, 3534, 525, 3668],
                        "render_bbox": [321, 3567, 503, 3634],
                        "safe_text_box": [318, 3554, 506, 3648],
                        "fit_status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "debug": {"root": str(debug_root)},
                "paginas": [
                    {
                        "numero": 2,
                        "text_layers": [
                            {
                                "id": "tl_002_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_002_band_019",
                                "band_id": "page_002_band_019",
                                "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                                "source_bbox": [235, 3470, 548, 3602],
                                "text_pixel_bbox": [235, 3470, 548, 3602],
                                "balloon_bbox": [0, 3427, 800, 3755],
                            },
                            {
                                "id": "tl_002_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_002_band_019",
                                "band_id": "page_002_band_019",
                                "translated": "ATRIZ...",
                                "source_bbox": [324, 3622, 508, 3668],
                                "text_pixel_bbox": [324, 3622, 508, 3668],
                                "balloon_bbox": [324, 3622, 508, 3668],
                            },
                        ],
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

        first, second = project["paginas"][0]["text_layers"]
        self.assertEqual(audit["hydrated_layers"], 1)
        self.assertTrue(first.get("visible", True))
        self.assertEqual(first["translated"], "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...")
        self.assertEqual(first["source_trace_ids"], ["ocr_002@page_002_band_019", "ocr_001@page_002_band_019"])
        self.assertEqual(first["render_bbox"], [321, 3567, 503, 3634])
        self.assertFalse(second.get("visible", True))
        self.assertEqual(second.get("merged_into_trace_id"), "ocr_001@page_002_band_019")

    def test_hydrate_project_render_metadata_offsets_group_candidate_and_hides_child_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_004_band_054",
                        "text_id": "ocr_001",
                        "source_text_ids": ["ocr_001", "ocr_002"],
                        "source_trace_ids": [
                            "ocr_001@page_004_band_054",
                            "ocr_002@page_004_band_054",
                        ],
                        "band_id": "page_004_band_054",
                        "page_id": "page_004",
                        "translated": "OPPA,\nPOR QUE VOCÊ ESTÁ SOLITÁRIO?",
                        "target_bbox": [0, 60, 425, 391],
                        "safe_text_box": [92, 131, 377, 244],
                        "render_bbox": [114, 162, 355, 213],
                        "fit_status": "ok",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "debug": {"root": str(debug_root)},
                "paginas": [
                    {
                        "numero": 4,
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_004_band_054",
                                "band_id": "page_004_band_054",
                                "page_id": "page_004",
                                "translated": "OPPA,",
                                "bbox": [0, 4383, 425, 4714],
                                "source_bbox": [0, 4383, 425, 4714],
                                "text_pixel_bbox": [0, 4383, 425, 4714],
                                "target_bbox": [0, 4383, 425, 4714],
                                "safe_text_box": [89, 172, 329, 322],
                                "render_bbox": [89, 172, 329, 322],
                                "qa_flags": ["safe_text_box_recomputed"],
                            },
                            {
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_004_band_054",
                                "band_id": "page_004_band_054",
                                "page_id": "page_004",
                                "translated": "POR QUE VOCÊ ESTÁ SOLITÁRIO?",
                                "bbox": [92, 4454, 377, 4567],
                                "source_bbox": [92, 4454, 377, 4567],
                                "text_pixel_bbox": [92, 4454, 377, 4567],
                                "qa_flags": ["missing_render_bbox"],
                            },
                        ],
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            contract = main._ensure_project_render_contract(project)

        first, second = project["paginas"][0]["text_layers"]
        self.assertGreaterEqual(audit["hydrated_layers"], 1)
        self.assertTrue(first.get("visible", True))
        self.assertEqual(first["translated"], "OPPA,\nPOR QUE VOCÊ ESTÁ SOLITÁRIO?")
        self.assertEqual(first["safe_text_box"], [92, 4454, 377, 4567])
        self.assertEqual(first["render_bbox"], [114, 4485, 355, 4536])
        self.assertNotIn("missing_render_bbox", first.get("qa_flags") or [])
        self.assertFalse(second.get("visible", True))
        self.assertEqual(second.get("merged_into_trace_id"), "ocr_001@page_004_band_054")
        self.assertEqual(contract["missing_render_bbox_count"], 0)

    def test_hydrate_project_render_metadata_replaces_existing_local_render_when_target_is_page_space(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            debug_root = work_dir / "debug" / "e2e"
            (debug_root / "09_typeset").mkdir(parents=True)
            (debug_root / "09_typeset" / "render_plan_raw.jsonl").write_text(
                json.dumps(
                    {
                        "trace_id": "ocr_001@page_004_band_054",
                        "text_id": "ocr_001",
                        "source_text_ids": ["ocr_001", "ocr_002"],
                        "source_trace_ids": [
                            "ocr_001@page_004_band_054",
                            "ocr_002@page_004_band_054",
                        ],
                        "band_id": "page_004_band_054",
                        "page_id": "page_004",
                        "translated": "OPPA,\nPOR QUE VOCÊ ESTÁ SOLITÁRIO?",
                        "target_bbox": [0, 60, 425, 391],
                        "safe_text_box": [92, 131, 377, 244],
                        "render_bbox": [114, 162, 355, 213],
                        "fit_status": "ok",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "debug": {"root": str(debug_root)},
                "paginas": [
                    {
                        "numero": 4,
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_004_band_054",
                                "band_id": "page_004_band_054",
                                "page_id": "page_004",
                                "translated": "OPPA,",
                                "bbox": [120, 191, 294, 276],
                                "source_bbox": [120, 191, 294, 276],
                                "text_pixel_bbox": [120, 191, 294, 276],
                                "target_bbox": [0, 4383, 425, 4714],
                                "safe_text_box": [89, 172, 329, 322],
                                "render_bbox": [89, 172, 329, 322],
                                "_render_metadata_hydrated": True,
                                "qa_flags": ["safe_text_box_recomputed"],
                            },
                            {
                                "id": "ocr_002",
                                "text_id": "ocr_002",
                                "trace_id": "ocr_002@page_004_band_054",
                                "band_id": "page_004_band_054",
                                "page_id": "page_004",
                                "translated": "POR QUE VOCÊ ESTÁ SOLITÁRIO?",
                                "bbox": [92, 4454, 377, 4567],
                                "source_bbox": [92, 4454, 377, 4567],
                                "text_pixel_bbox": [92, 4454, 377, 4567],
                            },
                        ],
                    }
                ],
            }

            main._hydrate_project_render_metadata_from_debug_candidates(project)
            main._merge_same_balloon_fragment_layers(project)

        first, second = project["paginas"][0]["text_layers"]
        self.assertEqual(first["translated"], "OPPA,\nPOR QUE VOCÊ ESTÁ SOLITÁRIO?")
        self.assertEqual(first["safe_text_box"], [92, 4454, 377, 4567])
        self.assertEqual(first["render_bbox"], [114, 4485, 355, 4536])
        self.assertFalse(second.get("visible", True))

    def test_merge_same_balloon_fragment_layers_uses_existing_source_ids_without_flag(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 4,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_004_band_054",
                            "band_id": "page_004_band_054",
                            "translated": "OPPA,",
                            "bbox": [120, 4514, 294, 4599],
                            "text_pixel_bbox": [120, 4514, 294, 4599],
                            "target_bbox": [0, 4383, 425, 4714],
                            "safe_text_box": [89, 4476, 329, 4626],
                            "render_bbox": [89, 4476, 329, 4626],
                            "source_text_ids": [
                                "ocr_001",
                                "ocr_002",
                                "ocr_001@page_004_band_054",
                                "ocr_002@page_004_band_054",
                            ],
                        },
                        {
                            "id": "ocr_002",
                            "text_id": "ocr_002",
                            "trace_id": "ocr_002@page_004_band_054",
                            "band_id": "page_004_band_054",
                            "translated": "POR QUE VOCÊ ESTÁ SOLITÁRIO?",
                            "bbox": [120, 4546, 294, 4599],
                            "text_pixel_bbox": [120, 4546, 294, 4599],
                            "target_bbox": [0, 4383, 425, 4714],
                            "safe_text_box": [89, 4476, 329, 4626],
                            "render_bbox": [89, 4476, 329, 4626],
                        },
                    ],
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        first, second = project["paginas"][0]["text_layers"]
        self.assertEqual(merged, 1)
        self.assertEqual(first["translated"], "OPPA,\nPOR QUE VOCÊ ESTÁ SOLITÁRIO?")
        self.assertFalse(second.get("visible", True))
        self.assertEqual(second.get("render_policy"), "merged_into_primary")

    def test_merge_same_balloon_fragment_layers_folds_hidden_source_text(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_004",
                            "text_id": "ocr_004",
                            "trace_id": "ocr_004@page_002_band_007",
                            "band_id": "page_002_band_007",
                            "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES",
                            "bbox": [527, 7113, 688, 7221],
                            "text_pixel_bbox": [527, 7113, 688, 7221],
                            "target_bbox": [461, 7040, 754, 7275],
                            "safe_text_box": [550, 7095, 668, 7228],
                            "render_bbox": [559, 7100, 660, 7222],
                            "source_text_ids": [
                                "ocr_004",
                                "ocr_003",
                                "ocr_004@page_002_band_007",
                                "ocr_003@page_002_band_007",
                            ],
                            "source_trace_ids": [
                                "ocr_004@page_002_band_007",
                                "ocr_003@page_002_band_007",
                            ],
                            "qa_flags": ["same_balloon_fragment_merged"],
                        },
                        {
                            "id": "ocr_003",
                            "text_id": "ocr_003",
                            "trace_id": "ocr_003@page_002_band_007",
                            "band_id": "page_002_band_007",
                            "translated": "O DIRETOR",
                            "bbox": [555, 7209, 661, 7221],
                            "text_pixel_bbox": [555, 7209, 661, 7221],
                            "visible": False,
                            "merged_into_trace_id": "ocr_004@page_002_band_007",
                        },
                    ],
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        primary, hidden = project["paginas"][0]["text_layers"]
        self.assertEqual(merged, 1)
        self.assertEqual(
            primary["translated"],
            "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES\nO DIRETOR",
        )
        self.assertFalse(hidden.get("visible", True))

    def test_merge_same_balloon_fragment_layers_ignores_already_merged_hidden_text(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 5,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_005_band_078",
                            "band_id": "page_005_band_078",
                            "translated": "A retenção do subespaço é de apenas cinco minutos.",
                            "bbox": [131, 112, 312, 231],
                            "text_pixel_bbox": [131, 112, 312, 231],
                            "target_bbox": [56, 28, 713, 593],
                            "source_text_ids": ["ocr_001", "ocr_001_002"],
                            "source_trace_ids": [
                                "ocr_001@page_005_band_078",
                                "ocr_001_002@page_005_band_078",
                            ],
                            "qa_flags": ["same_balloon_fragment_merged"],
                        },
                        {
                            "id": "ocr_001_002",
                            "text_id": "ocr_001_002",
                            "trace_id": "ocr_001_002@page_005_band_078",
                            "band_id": "page_005_band_078",
                            "translated": "Space is only utes. se você ultrapassar esse tempo.",
                            "bbox": [399, 270, 675, 401],
                            "text_pixel_bbox": [399, 270, 675, 401],
                            "visible": False,
                            "route_action": "merged_into_primary",
                            "render_policy": "merged_into_primary",
                            "merged_into_trace_id": "ocr_001@page_005_band_078",
                        },
                    ],
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        primary, hidden = project["paginas"][0]["text_layers"]
        self.assertEqual(merged, 0)
        self.assertEqual(primary["translated"], "A retenção do subespaço é de apenas cinco minutos.")
        self.assertFalse(hidden.get("visible", True))

    def test_merge_same_balloon_fragment_layers_keeps_dark_lobes_separate_with_stale_text_pixel_bbox(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 5,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_005_band_078",
                            "band_id": "page_005_band_078",
                            "visible": True,
                            "translated": "A retencao do subespaco e de apenas cinco minutos",
                            "text_pixel_bbox": [131, 8432, 312, 8551],
                            "bbox": [123, 8416, 318, 8559],
                            "layout_bbox": [131, 8432, 312, 8551],
                            "balloon_bbox": [16, 8352, 425, 8752],
                            "bubble_mask_bbox": [56, 8348, 713, 8913],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "qa_flags": ["dark_bubble_oval_reocr"],
                        },
                        {
                            "id": "ocr_001_002",
                            "text_id": "ocr_001_002",
                            "trace_id": "ocr_001_002@page_005_band_078",
                            "band_id": "page_005_band_078",
                            "visible": True,
                            "translated": "Se voce ultrapassar esse tempo, retornara ao seu mundo original!",
                            "source_trace_ids": [
                                "ocr_001_002@page_005_band_078",
                                "ocr_001@page_005_band_078",
                            ],
                            "source_text_ids": ["ocr_001_002", "ocr_001"],
                            "text_pixel_bbox": [131, 8432, 312, 8551],
                            "bbox": [399, 8524, 675, 8655],
                            "layout_bbox": [399, 8524, 675, 8655],
                            "balloon_bbox": [16, 8352, 425, 8752],
                            "bubble_mask_bbox": [56, 8348, 713, 8913],
                            "bubble_mask_source": "image_dark_bubble_mask",
                            "qa_flags": ["dark_bubble_oval_reocr", "same_balloon_fragment_merged"],
                        },
                    ],
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        left, right = project["paginas"][0]["text_layers"]
        self.assertEqual(merged, 0)
        self.assertIsNot(left.get("visible"), False)
        self.assertIsNot(right.get("visible"), False)
        self.assertNotEqual(right.get("render_policy"), "merged_into_primary")
        self.assertNotIn("retencao do subespaco", right["translated"])

    def test_repair_distinct_dark_lobe_project_payload_merges_restores_sibling_translation(self) -> None:
        project_layers = [
            {
                "id": "ocr_001",
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_005_band_078",
                "band_id": "page_005_band_078",
                "visible": True,
                "text": "The subspace retention is only five minutes",
                "translated": (
                    "A retencao do subespaco e de apenas cinco minutos "
                    "Space is only utes. se voce ultrapassar esse tempo, voce retornara ao seu mundo original!"
                ),
                "traduzido": (
                    "A retencao do subespaco e de apenas cinco minutos "
                    "Space is only utes. se voce ultrapassar esse tempo, voce retornara ao seu mundo original!"
                ),
                "source_trace_ids": ["ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"],
                "_source_trace_ids": ["ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"],
                "source_text_ids": ["ocr_001", "ocr_001_002", "ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"],
                "_source_text_ids": ["ocr_001", "ocr_001_002", "ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"],
                "text_pixel_bbox": [131, 8432, 312, 8551],
                "bbox": [131, 8432, 312, 8551],
                "layout_bbox": [131, 8432, 312, 8551],
                "balloon_bbox": [16, 8352, 425, 8752],
                "bubble_mask_bbox": [56, 8348, 713, 8913],
                "safe_text_box": [148, 8431, 620, 8830],
                "render_bbox": [180, 8499, 588, 8761],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": ["dark_bubble_oval_reocr", "same_balloon_fragment_merged"],
            },
            {
                "id": "ocr_001_002",
                "text_id": "ocr_001_002",
                "trace_id": "ocr_001_002@page_005_band_078",
                "band_id": "page_005_band_078",
                "visible": True,
                "text": "If you exceed that time, you will return to your original world!",
                "translated": "",
                "traduzido": "",
                "text_pixel_bbox": [399, 8524, 675, 8655],
                "bbox": [399, 8524, 675, 8655],
                "layout_bbox": [399, 8524, 675, 8655],
                "balloon_bbox": [16, 8352, 425, 8752],
                "bubble_mask_bbox": [56, 8348, 713, 8913],
                "safe_text_box": [148, 8431, 620, 8830],
                "render_bbox": [180, 8499, 588, 8761],
                "bubble_mask_source": "image_dark_bubble_mask",
                "qa_flags": [
                    "dark_bubble_oval_reocr",
                    "leading_dark_lobe_duplicate_fragment_removed",
                    "stale_text_pixel_bbox_repaired",
                ],
                "qa_metrics": {
                    "leading_dark_lobe_duplicate_fragment_removed": {
                        "from": "space is only utes. If you exceed that time, you will return to your original world!",
                        "to": "If you exceed that time, you will return to your original world!",
                        "matched_text_id": "ocr_001",
                    },
                },
            },
        ]

        repaired = main._repair_distinct_dark_lobe_project_payload_merges(project_layers)

        left, right = project_layers
        self.assertEqual(repaired, 1)
        self.assertEqual(left["translated"], "A retencao do subespaco e de apenas cinco minutos")
        self.assertEqual(right["translated"], "se voce ultrapassar esse tempo, voce retornara ao seu mundo original")
        self.assertEqual(left["source_trace_ids"], ["ocr_001@page_005_band_078"])
        self.assertEqual(right["source_trace_ids"], ["ocr_001_002@page_005_band_078"])
        self.assertNotIn("same_balloon_fragment_merged", left["qa_flags"])
        self.assertNotIn("same_balloon_fragment_merged", right["qa_flags"])
        self.assertIn("distinct_dark_lobe_payload_merge_repaired", left["qa_flags"])
        self.assertIn("distinct_dark_lobe_payload_merge_repaired", right["qa_flags"])
        self.assertEqual(left["safe_text_box"], [148, 8431, 373, 8830])
        self.assertEqual(right["safe_text_box"], [385, 8431, 620, 8830])
        self.assertEqual(left["render_bbox"], [148, 8432, 312, 8551])
        self.assertEqual(right["render_bbox"], [399, 8524, 620, 8655])

        left["translated"] = (
            "A retencao do subespaco e de apenas cinco minutos "
            "Space is only utes. se voce ultrapassar esse tempo, voce retornara ao seu mundo original!"
        )
        left["traduzido"] = left["translated"]
        left["source_trace_ids"] = ["ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"]
        left["_source_trace_ids"] = ["ocr_001@page_005_band_078", "ocr_001_002@page_005_band_078"]
        left["source_text_ids"] = ["ocr_001", "ocr_001_002"]
        left["_source_text_ids"] = ["ocr_001", "ocr_001_002"]
        left["qa_flags"].append("same_balloon_fragment_merged")
        left["render_bbox"] = [180, 8499, 588, 8761]
        right["render_bbox"] = [180, 8499, 588, 8761]

        repaired_again = main._repair_distinct_dark_lobe_project_payload_merges(project_layers)

        self.assertEqual(repaired_again, 1)
        self.assertEqual(left["translated"], "A retencao do subespaco e de apenas cinco minutos")
        self.assertEqual(right["translated"], "se voce ultrapassar esse tempo, voce retornara ao seu mundo original")
        self.assertNotIn("same_balloon_fragment_merged", left["qa_flags"])
        self.assertNotIn("same_balloon_fragment_merged", right["qa_flags"])

        left.pop("safe_text_box", None)
        left.pop("_debug_safe_text_box", None)
        left.pop("render_bbox", None)
        left.pop("position_bbox", None)
        left.pop("capacity_bbox", None)
        left["layout_safe_bbox"] = [148, 8431, 373, 8830]
        left["qa_flags"].extend(["missing_render_bbox", "same_balloon_fragment_merged"])
        right.pop("safe_text_box", None)
        right.pop("_debug_safe_text_box", None)
        right.pop("render_bbox", None)
        right.pop("position_bbox", None)
        right.pop("capacity_bbox", None)
        right.pop("layout_safe_bbox", None)
        right["qa_flags"].extend(["missing_render_bbox", "same_balloon_fragment_merged"])

        finalized = main._finalize_distinct_dark_lobe_project_geometry(project_layers)

        self.assertGreaterEqual(finalized, 1)
        self.assertEqual(left["safe_text_box"], [148, 8431, 373, 8830])
        self.assertEqual(left["render_bbox"], [148, 8432, 312, 8551])
        self.assertEqual(right["safe_text_box"], [424, 8431, 649, 8830])
        self.assertEqual(right["render_bbox"], [424, 8524, 649, 8655])
        self.assertNotIn("missing_render_bbox", left["qa_flags"])
        self.assertNotIn("same_balloon_fragment_merged", left["qa_flags"])
        self.assertNotIn("missing_render_bbox", right["qa_flags"])
        self.assertNotIn("same_balloon_fragment_merged", right["qa_flags"])

    def test_merge_same_balloon_fragment_layers_collapses_cross_page_band_siblings(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                            "balloon_bbox": [0, 16180, 800, 16383],
                            "safe_text_box": [318, 16293, 504, 16383],
                            "qa_flags": ["same_balloon_fragment_merged"],
                        }
                    ],
                },
                {
                    "numero": 3,
                    "text_layers": [
                        {
                            "id": "ocr_002",
                            "text_id": "ocr_002",
                            "trace_id": "ocr_002@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...",
                            "source_trace_ids": [
                                "ocr_002@page_002_band_019",
                                "ocr_001@page_002_band_019",
                            ],
                            "source_text_ids": ["ocr_002", "ocr_001"],
                            "balloon_bbox": [324, 0, 508, 38],
                            "safe_text_box": [318, 0, 504, 29],
                            "qa_flags": ["same_balloon_fragment_merged"],
                        }
                    ],
                },
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        primary = project["paginas"][0]["text_layers"][0]
        sibling = project["paginas"][1]["text_layers"][0]
        self.assertEqual(merged, 1)
        self.assertTrue(primary.get("visible", True))
        self.assertEqual(primary["translated"], "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL ATRIZ...")
        self.assertFalse(sibling.get("visible", True))
        self.assertEqual(sibling.get("merged_into_trace_id"), "ocr_001@page_002_band_019")
        self.assertEqual(
            primary.get("source_trace_ids"),
            ["ocr_002@page_002_band_019", "ocr_001@page_002_band_019"],
        )

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

    def test_debug_qa_propagation_drops_suppressed_low_containment_flag_for_clean_layer(self) -> None:
        project = {
            "_work_dir": "dummy",
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_003_band_035",
                            "route_action": "translate_inpaint_render",
                            "translated": "EI, VAMOS!",
                            "render_bbox": [148, 235, 300, 260],
                            "safe_text_box": [85, 219, 393, 271],
                            "fit_status": "ok",
                            "qa_flags": [],
                        }
                    ]
                }
            ],
        }
        claim = {
            "identity_groups": [["ocr_001@page_003_band_035", "ocr_001"]],
            "flags": {"render_suppressed_low_containment_fragment"},
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

    def test_debug_qa_propagation_marks_fast_fill_no_glyph_evidence_as_review_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            render_plan = (
                tmp_root / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
            )
            render_plan.parent.mkdir(parents=True, exist_ok=True)
            render_plan.write_text(
                json.dumps(
                    {
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_051_band_127",
                        "band_id": "page_051_band_127",
                        "qa_flags": ["fast_fill_no_glyph_evidence"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            inpaint_root = (
                tmp_root / "debug" / "e2e" / "08_inpaint" / "page_051_band_127"
            )
            inpaint_root.mkdir(parents=True, exist_ok=True)
            (inpaint_root / "inpaint_decision.json").write_text(
                json.dumps(
                    {
                        "band_id": "page_051_band_127",
                        "trace_ids": ["ocr_003@page_051_band_127"],
                        "flags": ["fast_fill_no_glyph_evidence"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            project = {
                "_work_dir": str(tmp_root),
                "paginas": [{"numero": 51, "text_layers": []}],
                "qa": {},
            }
            audit = main._propagate_debug_qa_flags_to_project(project)

            self.assertEqual(audit["summary"]["qa_flag_not_propagated_count"], 2)
            self.assertTrue(all(item["is_review_only"] for item in audit["missing_in_project"]))

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

    def test_final_page_space_text_layers_drop_degenerate_target_box_before_rerender(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "ocr_001",
                        "original": "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL",
                        "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                        "bbox": [298, 3534, 525, 3621],
                        "source_bbox": [298, 3523, 533, 3630],
                        "balloon_bbox": [0, 3427, 800, 3755],
                        "target_bbox": [298, 3523, 525, 3657],
                        "safe_text_box": [318, 3543, 506, 3637],
                        "_debug_safe_text_box": [318, 3543, 506, 3637],
                        "render_bbox": [321, 3556, 503, 3623],
                        "qa_flags": [
                            "same_balloon_fragment_merged",
                            "rejected_derived_bubble_mask",
                            "tiny_bubble_inner_bbox_rejected",
                            "safe_text_box_recomputed",
                        ],
                        "qa_metrics": {"render_balloon_containment": 0.1762},
                        "route_action": "translate_inpaint_render",
                        "line_polygons": [
                            [[298, 3534], [525, 3534], [525, 3560], [298, 3560]],
                        ],
                    }
                ]
            },
            page_number=2,
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["balloon_bbox"], [0, 3427, 800, 3755])
        self.assertNotIn("target_bbox", normalized[0])
        self.assertNotIn("safe_text_box", normalized[0])
        self.assertNotIn("_debug_safe_text_box", normalized[0])
        self.assertNotIn("render_bbox", normalized[0])
        self.assertIn("safe_text_box_recomputed", normalized[0]["qa_flags"])

    def test_normalize_final_project_layers_drop_degenerate_target_box(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "original": "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL",
                            "translated": "A QUANTIA ESTÁ CERTA. ESSA VADIA É REAL",
                            "bbox": [298, 3534, 525, 3621],
                            "source_bbox": [298, 3523, 533, 3630],
                            "balloon_bbox": [0, 3427, 800, 3755],
                            "target_bbox": [298, 3523, 525, 3657],
                            "safe_text_box": [318, 3543, 506, 3637],
                            "_debug_safe_text_box": [318, 3543, 506, 3637],
                            "render_bbox": [321, 3556, 503, 3623],
                            "qa_flags": [
                                "same_balloon_fragment_merged",
                                "rejected_derived_bubble_mask",
                                "tiny_bubble_inner_bbox_rejected",
                                "safe_text_box_recomputed",
                            ],
                            "qa_metrics": {"render_balloon_containment": 0.1762},
                            "line_polygons": [
                                [[298, 3534], [525, 3534], [525, 3560], [298, 3560]],
                            ],
                        }
                    ],
                }
            ]
        }

        audit = main._normalize_final_project_page_space_layers(project)

        self.assertEqual(audit["layers_changed"], 1)
        layer = project["paginas"][0]["text_layers"][0]
        self.assertEqual(layer["balloon_bbox"], [0, 3427, 800, 3755])
        self.assertNotIn("target_bbox", layer)
        self.assertNotIn("safe_text_box", layer)
        self.assertNotIn("_debug_safe_text_box", layer)
        self.assertNotIn("render_bbox", layer)

    def test_final_project_page_space_suppresses_art_fragment_review_layers(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 6,
                    "text_layers": [
                        {
                            "id": "direct_paddle_reocr_001",
                            "trace_id": "direct_paddle_reocr_001@page_006_band_101",
                            "text": "WU",
                            "translated": "WU",
                            "bbox": [211, 3265, 319, 3375],
                            "text_pixel_bbox": [214, 3274, 249, 3308],
                            "route_action": "review_required",
                            "route_reason": "ocr_art_fragment_suspected",
                            "render_policy": "normal",
                            "visible": True,
                            "skip_processing": False,
                            "qa_flags": [
                                "candidate_crop_direct_paddle_reocr",
                                "dark_bubble_oval_reocr",
                                "ocr_art_fragment_suspected",
                            ],
                        }
                    ],
                }
            ]
        }

        audit = main._normalize_final_project_page_space_layers(project)

        self.assertEqual(audit["layers_changed"], 1)
        layer = project["paginas"][0]["text_layers"][0]
        self.assertFalse(layer["visible"])
        self.assertEqual(layer["render_policy"], "preserve_original")
        self.assertEqual(layer["route_action"], "review_required")
        self.assertTrue(layer["skip_processing"])

    def test_merge_same_balloon_fragments_derives_band_id_from_trace_id(self) -> None:
        project = {
            "paginas": [
                {
                    "numero": 2,
                    "text_layers": [
                        {
                            "id": "ocr_003",
                            "trace_id": "ocr_003@page_002_band_014",
                            "translated": "Afinal, é câncer, por que se preocupar?",
                            "bbox": [301, 12418, 690, 12551],
                            "text_pixel_bbox": [301, 12418, 690, 12551],
                            "balloon_bbox": [8, 11794, 749, 12573],
                            "bubble_mask_bbox": [8, 11794, 749, 12573],
                            "bubble_id": "page_002_band_014_bubble_003",
                            "route_action": "translate_inpaint_render",
                            "qa_flags": ["ocr_joined_repaired"],
                        },
                        {
                            "id": "ocr_005",
                            "trace_id": "ocr_005@page_002_band_014",
                            "translated": "Afinal, é câncer, por que se preocupar? sua vida é tão frustrante também",
                            "bbox": [301, 12418, 690, 12587],
                            "text_pixel_bbox": [301, 12418, 690, 12587],
                            "balloon_bbox": [209, 12347, 792, 12653],
                            "bubble_mask_bbox": [274, 12380, 727, 12620],
                            "bubble_id": "page_002_band_014_bubble_005",
                            "route_action": "translate_inpaint_render",
                            "source_text_ids": [
                                "ocr_005",
                                "ocr_003",
                                "ocr_003@page_002_band_014",
                                "ocr_005@page_002_band_014",
                            ],
                            "qa_flags": [
                                "ocr_joined_repaired",
                                "same_balloon_fragment_merged",
                                "debug_derived_bubble_mask_rejected",
                            ],
                        },
                    ],
                }
            ]
        }

        merged = main._merge_same_balloon_fragment_layers(project)

        self.assertEqual(merged, 1)
        first, second = project["paginas"][0]["text_layers"]
        self.assertFalse(first.get("visible", True))
        self.assertEqual(first.get("render_policy"), "merged_into_primary")
        self.assertNotEqual(second.get("render_policy"), "merged_into_primary")
        self.assertIn("frustrante", second["translated"].lower())

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

    def test_final_page_space_text_layers_shift_band_local_bubble_geometry(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "dialogue",
                        "original": "Please, for the child's sake.",
                        "translated": "Por favor, pela crianca.",
                        "band_id": "page_002_band_005",
                        "band_y_top": 4292,
                        "bbox": [509, 4607, 647, 4661],
                        "source_bbox": [501, 4559, 661, 4666],
                        "text_pixel_bbox": [509, 4607, 647, 4661],
                        "balloon_bbox": [465, 266, 696, 437],
                        "bubble_mask_bbox": [465, 266, 696, 437],
                        "bubble_inner_bbox": [501, 267, 661, 374],
                        "balloon_subregions": [[465, 266, 696, 437]],
                        "route_action": "translate_inpaint_render",
                    }
                ]
            },
            page_number=2,
        )

        layer = normalized[0]
        self.assertEqual(layer["bbox"], [509, 4607, 647, 4661])
        self.assertEqual(layer["balloon_bbox"], [465, 4558, 696, 4729])
        self.assertEqual(layer["bubble_mask_bbox"], [465, 4558, 696, 4729])
        self.assertEqual(layer["bubble_inner_bbox"], [501, 4559, 661, 4666])
        self.assertEqual(layer["balloon_subregions"], [[465, 4558, 696, 4729]])
        self.assertEqual(layer["coordinate_space"], "page")
        self.assertNotIn("band_y_top", layer)

    def test_final_page_space_text_layers_do_not_double_shift_page_geometry(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "dialogue",
                        "original": "More",
                        "translated": "Mais",
                        "band_id": "page_002_band_020",
                        "band_y_top": 13106,
                        "bbox": [540, 13103, 621, 13125],
                        "safe_text_box": [540, 13103, 621, 13125],
                        "render_bbox": [561, 13106, 601, 13122],
                        "balloon_bbox": [500, 96, 680, 260],
                        "route_action": "translate_inpaint_render",
                    }
                ]
            },
            page_number=2,
        )

        layer = normalized[0]
        self.assertEqual(layer["safe_text_box"], [540, 13103, 621, 13125])
        self.assertEqual(layer["render_bbox"], [561, 13106, 601, 13122])
        self.assertEqual(layer["balloon_bbox"], [500, 13202, 680, 13366])

    def test_final_page_space_text_layers_keep_page_bbox_but_lift_band_local_bubble(self) -> None:
        normalized = main._final_page_space_text_layers_for_renderer(
            {
                "texts": [
                    {
                        "id": "dialogue",
                        "original": "Please!",
                        "translated": "Por favor!",
                        "bbox": [114, 2647, 325, 2748],
                        "text_pixel_bbox": [114, 2647, 325, 2748],
                        "balloon_bbox": [64, 1075, 372, 1241],
                        "bubble_mask_bbox": [64, 1075, 372, 1241],
                        "band_y_top": 1572,
                        "strip_band_y_top": 17955,
                        "coordinate_space": "page",
                        "source_coordinate_space": "page",
                    }
                ]
            },
            page_number=2,
        )

        layer = normalized[0]
        self.assertEqual(layer["bbox"], [114, 2647, 325, 2748])
        self.assertEqual(layer["text_pixel_bbox"], [114, 2647, 325, 2748])
        self.assertEqual(layer["balloon_bbox"], [64, 2647, 372, 2813])
        self.assertEqual(layer["bubble_mask_bbox"], [64, 2647, 372, 2813])
        self.assertNotIn("band_y_top", layer)
        self.assertNotIn("strip_band_y_top", layer)

    def test_rerender_final_project_images_uses_final_project_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            (work_dir / "images").mkdir()
            (work_dir / "translated").mkdir()
            (work_dir / "images" / "002.jpg").write_bytes(b"fake")
            project = {
                "paginas": [
                    {
                        "numero": 2,
                        "arquivo_original": "originals/002.jpg",
                        "arquivo_traduzido": "translated/002.jpg",
                        "image_layers": {
                            "rendered": {
                                "key": "rendered",
                                "path": "translated/002.jpg",
                                "visible": True,
                                "locked": True,
                            }
                        },
                        "text_layers": [
                            {
                                "id": "ocr_002",
                                "translated": "POR FAVOR!",
                                "target_bbox": [419, 2735, 710, 2811],
                                "safe_text_box": [446, 2752, 657, 2797],
                            }
                        ],
                    }
                ]
            }

            with patch("main.render_page_image") as render_page:
                audit = main._rerender_final_project_images_from_metadata(project, work_dir)

            self.assertEqual(audit["pages_checked"], 1)
            self.assertEqual(audit["pages_rerendered"], 1)
            render_page.assert_called_once_with(
                project,
                0,
                str(work_dir / "translated" / "002.jpg"),
            )
            self.assertNotIn("_work_dir", project)

    def test_rerender_final_project_images_skips_strip_reassembled_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            reassemble_dir = work_dir / "debug" / "e2e" / "10_copyback_reassemble"
            reassemble_dir.mkdir(parents=True)
            (reassemble_dir / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_003_band_028",
                        "translated_output_page": "002.jpg",
                        "output_page_number": 2,
                        "band_y_top": 19737,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "paginas": [
                    {
                        "numero": 2,
                        "arquivo_original": "originals/002.jpg",
                        "arquivo_traduzido": "translated/002.jpg",
                        "image_layers": {
                            "rendered": {
                                "key": "rendered",
                                "path": "translated/002.jpg",
                                "visible": True,
                                "locked": True,
                            }
                        },
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "translated": "NINGUEM SE IMPORTAVA",
                                "band_id": "page_003_band_028",
                                "render_bbox": [106, 6806, 371, 6955],
                                "safe_text_box": [0, 6700, 400, 7100],
                            }
                        ],
                    }
                ]
            }

            with patch("main.render_page_image") as render_page:
                audit = main._rerender_final_project_images_from_metadata(project, work_dir)

            self.assertEqual(audit["pages_checked"], 0)
            self.assertEqual(audit["pages_rerendered"], 0)
            self.assertTrue(audit["skipped_strip_reassembled_output"])
            render_page.assert_not_called()
            self.assertNotIn("_work_dir", project)

    def test_final_project_rerender_allows_strip_output_when_late_geometry_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            (work_dir / "translated").mkdir(parents=True)
            (work_dir / "images").mkdir(parents=True)
            blank = np.zeros((20, 20, 3), dtype=np.uint8)
            cv2.imwrite(str(work_dir / "translated" / "002.jpg"), blank)
            cv2.imwrite(str(work_dir / "images" / "002.jpg"), blank)
            crops_dir = work_dir / "debug" / "e2e" / "10_copyback_reassemble"
            crops_dir.mkdir(parents=True)
            rendered_bands_dir = work_dir / "debug" / "e2e" / "09_typeset" / "rendered_bands"
            rendered_bands_dir.mkdir(parents=True)
            positive_band = np.zeros((20, 20, 3), dtype=np.uint8)
            positive_band[:, :] = (10, 20, 30)
            positive_band[3:8, 3:8, :] = (245, 245, 245)
            cv2.imwrite(str(rendered_bands_dir / "page_002_band_023.jpg"), positive_band)
            (crops_dir / "final_band_crops.jsonl").write_text(
                json.dumps(
                    {
                        "band_id": "page_002_band_023",
                        "translated_output_page": "002.jpg",
                        "output_page_number": 2,
                        "crop_bbox_in_translated_page": [0, 0, 20, 20],
                        "final_crop_path": "10_copyback_reassemble/final_bands/page_002_band_023.jpg",
                        "rendered_band_path": "09_typeset/rendered_bands/page_002_band_023.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            project = {
                "qa": {"post_style_component_safe_partition_count": 1},
                "paginas": [
                    {
                        "numero": 2,
                        "arquivo_original": "originals/002.jpg",
                        "arquivo_traduzido": "translated/002.jpg",
                        "image_layers": {
                            "rendered": {
                                "key": "rendered",
                                "path": "translated/002.jpg",
                                "visible": True,
                                "locked": True,
                            }
                        },
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "translated": "VOCE CRESCEU EM UM ORFANATO",
                                "band_id": "page_002_band_023",
                                "source_text_mask_bbox": [2, 2, 10, 10],
                                "render_bbox": [2, 2, 10, 10],
                                "safe_text_box": [2, 2, 10, 10],
                                "_render_bbox_from_repaired_safe_text_box": True,
                                "qa_flags": ["dark_connected_component_safe_partition"],
                            }
                        ],
                    }
                ],
            }

            rendered = np.full((20, 20, 3), 255, dtype=np.uint8)
            with patch("typesetter.renderer.render_band_image", return_value=rendered) as render_band:
                audit = main._rerender_final_project_images_from_metadata(project, work_dir)

            self.assertEqual(audit["pages_checked"], 1)
            self.assertEqual(audit["pages_rerendered"], 1)
            self.assertEqual(audit["rows_rerendered"], 1)
            self.assertEqual(audit["positive_band_base_used"], 1)
            self.assertGreaterEqual(audit["stale_text_regions_scrubbed"], 1)
            self.assertTrue(audit["strip_reassembled_output_rerender_allowed"])
            render_band.assert_called_once()
            self.assertFalse(np.any(render_band.call_args.args[0] == 245))
            self.assertTrue((work_dir / "debug" / "e2e" / "10_copyback_reassemble" / "final_bands" / "page_002_band_023.jpg").exists())
            self.assertNotIn("_work_dir", project)

    def test_debug_render_metadata_hydration_preserves_final_layout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            raw_entry = {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_002_band_003",
                "page_id": "page_002",
                "band_id": "page_002_band_003",
                "coordinate_space": "band",
                "band_y_top": 2732,
                "translated": "A MISSÃO PRINCIPAL SERÁ MOSTRADA EM BREVE",
                "target_bbox": [38, 122, 331, 270],
                "position_bbox": [52, 135, 319, 266],
                "safe_text_box": [52, 139, 319, 262],
                "render_bbox": [112, 153, 259, 247],
                "text_pixel_bbox": [113, 165, 259, 236],
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size_final": 26,
                "line_height": 36,
                "wrapped_lines": ["A MISSÃO PRINCIPAL", "SERÁ MOSTRADA", "EM BREVE"],
            }
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                json.dumps(raw_entry, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 2,
                        "text_layers": [
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_002_band_003",
                                "band_id": "page_002_band_003",
                                "translated": "A MISSÃO PRINCIPAL SERÁ MOSTRADA EM BREVE",
                                "bbox": [113, 2897, 259, 2968],
                                "text_pixel_bbox": [113, 2897, 259, 2968],
                                "source_bbox": [113, 2897, 259, 2968],
                            }
                        ],
                    }
                ],
            }

            audit = main._hydrate_project_render_metadata_from_debug_candidates(project)
            layer = project["paginas"][0]["text_layers"][0]
            normalized = main._final_page_space_text_layers_for_renderer([layer], page_number=2)[0]

            self.assertTrue(audit["hydrated_layers"] >= 1)
            self.assertEqual(layer["render_bbox"], [112, 2885, 259, 2979])
            self.assertEqual(layer["safe_text_box"], [52, 2871, 319, 2994])
            self.assertIn("render_layout_contract", layer)
            contract = layer["render_layout_contract"]
            self.assertEqual(contract["font_size"], 26)
            self.assertEqual(contract["line_height"], 36)
            self.assertEqual(contract["lines"], raw_entry["wrapped_lines"])
            self.assertEqual(contract["coordinate_space"], "page")
            self.assertEqual(contract["target_bbox"], [38, 2854, 331, 3002])
            self.assertEqual(normalized["render_layout_contract"], contract)
            self.assertEqual(normalized["style"]["fonte"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
            self.assertEqual(normalized["style"]["tamanho"], 26)
            final_row = main._project_render_plan_row({"numero": 2}, normalized, 0)
            self.assertEqual(final_row["font_name"], "LeagueGothic-Regular-VariableFont_wdth.ttf")
            self.assertEqual(final_row["font_size_final"], 26)
            self.assertEqual(final_row["line_height"], 36)
            self.assertEqual(final_row["wrapped_lines"], raw_entry["wrapped_lines"])
            self.assertEqual(final_row["render_layout_contract"], contract)

    def test_debug_render_metadata_prefers_page_space_candidate_over_strip_band_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            band_entry = {
                "text_id": "direct_paddle_reocr_001",
                "trace_id": "direct_paddle_reocr_001@page_005_band_078",
                "band_id": "page_005_band_078",
                "coordinate_space": "band",
                "band_y_top": 60522,
                "translated": "A RETENÇÃO DO SUBESPAÇO É DE APENAS CINCO MINUTOS.",
                "target_bbox": [0, 0, 400, 700],
                "safe_text_box": [104, 78, 364, 645],
                "render_bbox": [113, 78, 327, 301],
                "text_pixel_bbox": [237, 139, 677, 355],
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size_final": 48,
                "line_height": 62,
                "wrapped_lines": ["A RETENÇÃO DO", "SUBESPAÇO É DE", "APENAS CINCO", "MINUTOS."],
            }
            page_entry = {
                **band_entry,
                "coordinate_space": "page",
                "band_y_top": 0,
                "safe_text_box": [27, 8379, 358, 9098],
                "render_bbox": [113, 8381, 327, 8604],
                "target_bbox": [0, 8200, 400, 9200],
                "text_pixel_bbox": [237, 8439, 677, 8655],
            }
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                json.dumps(band_entry, ensure_ascii=False) + "\n"
                + json.dumps(page_entry, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 5,
                        "text_layers": [
                            {
                                "id": "direct_paddle_reocr_001",
                                "text_id": "direct_paddle_reocr_001",
                                "trace_id": "direct_paddle_reocr_001@page_005_band_078",
                                "band_id": "page_005_band_078",
                                "translated": "A RETENÇÃO DO SUBESPAÇO É DE APENAS CINCO MINUTOS.",
                                "bbox": [237, 8439, 677, 8655],
                                "text_pixel_bbox": [237, 8439, 677, 8655],
                                "source_bbox": [237, 8439, 677, 8655],
                            }
                        ],
                    }
                ],
            }

            main._hydrate_project_render_metadata_from_debug_candidates(project)
            layer = project["paginas"][0]["text_layers"][0]

            self.assertEqual(layer["render_bbox"], [113, 8381, 327, 8604])
            self.assertEqual(layer["safe_text_box"], [27, 8379, 358, 9098])
            self.assertEqual(layer["render_layout_contract"]["coordinate_space"], "page")
            self.assertEqual(layer["render_layout_contract"]["band_y_top"], 0)

    def test_debug_render_metadata_does_not_apply_same_band_offset_to_page_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            typeset_dir = work_dir / "debug" / "e2e" / "09_typeset"
            typeset_dir.mkdir(parents=True)
            band_entry = {
                "text_id": "direct_paddle_reocr_001",
                "trace_id": "direct_paddle_reocr_001@page_005_band_078",
                "band_id": "page_005_band_078",
                "coordinate_space": "band",
                "band_y_top": 60522,
                "translated": "A RETENCAO DO SUBESPACO E DE APENAS CINCO MINUTOS.",
                "target_bbox": [0, 0, 400, 700],
                "safe_text_box": [104, 78, 364, 645],
                "render_bbox": [113, 78, 327, 301],
                "text_pixel_bbox": [237, 139, 677, 355],
                "font_name": "LeagueGothic-Regular-VariableFont_wdth.ttf",
                "font_size_final": 48,
                "line_height": 62,
                "wrapped_lines": ["A RETENCAO DO", "SUBESPACO E DE", "APENAS CINCO", "MINUTOS."],
            }
            page_entry = {
                **band_entry,
                "coordinate_space": "page",
                "band_y_top": 0,
                "target_bbox": [0, 8320, 385, 9157],
                "safe_text_box": [27, 8379, 358, 9098],
                "render_bbox": [113, 8381, 327, 8604],
                "text_pixel_bbox": [237, 8431, 677, 8554],
            }
            sibling_band_entry = {
                **band_entry,
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_005_band_078",
                "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
                "target_bbox": [385, 32, 744, 691],
                "safe_text_box": [198, 376, 755, 825],
                "render_bbox": [271, 376, 642, 533],
                "text_pixel_bbox": [399, 202, 675, 333],
                "wrapped_lines": ["SE VOCE ULTRAPASSAR ESSE", "TEMPO, VOCE RETORNARA", "AO SEU MUNDO ORIGINAL!"],
            }
            sibling_page_entry = {
                **sibling_band_entry,
                "coordinate_space": "page",
                "band_y_top": 0,
                "target_bbox": [385, 8320, 800, 9157],
                "safe_text_box": [198, 60898, 755, 61347],
                "render_bbox": [271, 60898, 642, 61055],
                "text_pixel_bbox": [399, 8524, 675, 8655],
            }
            (typeset_dir / "render_plan_raw.jsonl").write_text(
                "\n".join(json.dumps(entry, ensure_ascii=False) for entry in (
                    band_entry,
                    page_entry,
                    sibling_band_entry,
                    sibling_page_entry,
                ))
                + "\n",
                encoding="utf-8",
            )
            project = {
                "_work_dir": str(work_dir),
                "paginas": [
                    {
                        "numero": 5,
                        "text_layers": [
                            {
                                "id": "direct_paddle_reocr_001",
                                "text_id": "direct_paddle_reocr_001",
                                "trace_id": "direct_paddle_reocr_001@page_005_band_078",
                                "band_id": "page_005_band_078",
                                "translated": "A RETENCAO DO SUBESPACO E DE APENAS CINCO MINUTOS.",
                                "bbox": [237, 8431, 677, 8554],
                                "text_pixel_bbox": [237, 8431, 677, 8554],
                                "source_bbox": [237, 8431, 677, 8554],
                                "target_bbox": [0, 8320, 385, 9157],
                            },
                            {
                                "id": "ocr_001",
                                "text_id": "ocr_001",
                                "trace_id": "ocr_001@page_005_band_078",
                                "band_id": "page_005_band_078",
                                "translated": "SE VOCE ULTRAPASSAR ESSE TEMPO, VOCE RETORNARA AO SEU MUNDO ORIGINAL!",
                                "bbox": [399, 8524, 675, 8655],
                                "text_pixel_bbox": [399, 8524, 675, 8655],
                                "source_bbox": [399, 8524, 675, 8655],
                                "target_bbox": [385, 8320, 800, 9157],
                            },
                        ],
                    }
                ],
            }

            main._hydrate_project_render_metadata_from_debug_candidates(project)
            layer = project["paginas"][0]["text_layers"][0]

            self.assertEqual(layer["render_bbox"], [113, 8381, 327, 8604])
            self.assertEqual(layer["safe_text_box"], [27, 8379, 358, 9098])
            self.assertEqual(layer["render_layout_contract"]["coordinate_space"], "page")
            self.assertLess(layer["render_layout_contract"]["block_bbox"][1], 9000)
            self.assertNotEqual(layer["render_bbox"][1], 68903)

    def test_persist_real_bubble_mask_layer_injects_renderer_contract(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            mask_a = np.zeros((6, 8), dtype=np.uint8)
            mask_a[1:4, 1:5] = 255
            mask_b = np.zeros((6, 8), dtype=np.uint8)
            mask_b[2:6, 5:8] = 255
            page = {
                "numero": 1,
                "image_layers": {},
                "text_layers": [
                    {
                        "id": "dialogue",
                        "bubble_id": "worker_bubble_001",
                        "bbox": [1, 1, 5, 4],
                    }
                ],
            }
            ocr_page = {
                "_bubble_regions": [
                    {"bubble_id": "worker_bubble_001", "bubble_mask": mask_a},
                    {"bubble_id": "worker_bubble_002", "bubble_mask": mask_b},
                ]
            }

            result = main._persist_real_bubble_mask_layer_for_page(
                page,
                ocr_page,
                work_dir,
                page_number=1,
                image_size=(8, 6),
            )

            mask_path = work_dir / "layers" / "bubble-mask" / "001.png"
            self.assertTrue(result)
            self.assertTrue(mask_path.exists())
            self.assertEqual(page["image_layers"]["bubble_mask"]["path"], "layers/bubble-mask/001.png")
            self.assertEqual(page["text_layers"][0]["bubble_mask_path"], str(mask_path))
            self.assertEqual(page["text_layers"][0]["bubble_mask_value"], 1)
            saved = np.array(Image.open(mask_path).convert("L"))
            self.assertEqual(int(saved[2, 2]), 1)
            self.assertEqual(int(saved[3, 6]), 2)

    def test_build_project_json_persists_bubble_mask_without_serializing_ndarray(self) -> None:
        import json
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            image_path = work_dir / "001.jpg"
            Image.new("RGB", (32, 40), "white").save(image_path)
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[1:5, 2:7] = 255
            config = {
                "work_dir": str(work_dir),
                "obra": "One Second",
                "capitulo": 1,
                "idioma_origem": "en",
                "idioma_destino": "pt-BR",
            }
            ocr_results = [
                {
                    "_bubble_regions": [
                        {
                            "bubble_id": "bubble_001",
                            "bubble_mask": mask,
                            "bubble_mask_bbox": [10, 20, 18, 26],
                        }
                    ],
                    "_vision_blocks": [
                        {
                            "bbox": [10, 20, 18, 26],
                            "bubble_id": "bubble_001",
                            "bubble_mask": mask,
                            "bubble_mask_bbox": [10, 20, 18, 26],
                        }
                    ],
                }
            ]
            page_text_layers = [
                {
                    "texts": [
                        {
                            "id": "ocr_001",
                            "text": "Hello",
                            "translated": "Olá",
                            "bbox": [11, 21, 17, 25],
                            "source_bbox": [11, 21, 17, 25],
                            "bubble_id": "bubble_001",
                            "bubble_mask": mask,
                            "bubble_mask_bbox": [10, 20, 18, 26],
                        }
                    ]
                }
            ]

            project = main.build_project_json(
                config,
                {},
                ocr_results,
                page_text_layers,
                [image_path],
                1,
                0.1,
            )

            json.dumps(project)
            page = project["paginas"][0]
            layer = page["text_layers"][0]
            legacy = page["textos"][0]
            self.assertIn("bubble_mask", page["image_layers"])
            self.assertNotIn("bubble_mask", layer)
            self.assertNotIn("bubble_mask", legacy)
            self.assertEqual(layer["bubble_mask_layer_path"], "layers/bubble-mask/001.png")
            self.assertEqual(layer["bubble_mask_value"], 1)
            saved = np.array(Image.open(work_dir / "layers" / "bubble-mask" / "001.png").convert("L"))
            self.assertEqual(int(saved[22, 13]), 1)

    def test_normalize_text_layer_for_renderer_neutralizes_legacy_decision_fields(self) -> None:
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
                "skip_processing": True,
                "preserve_original": True,
                "render_policy": "preserve_original",
            },
            3,
            12,
        )

        self.assertEqual(layer["tipo"], "text")
        self.assertEqual(layer["content_class"], "text")
        self.assertEqual(layer["balloon_type"], "")
        self.assertFalse(layer["skip_processing"])
        self.assertFalse(layer["preserve_original"])
        self.assertEqual(layer["render_policy"], "normal")
        self.assertEqual(layer["route_action"], "translate_inpaint_render")

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
