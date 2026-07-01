import json
from pathlib import Path

from debug_tools import DebugRecorder
from qa.export_gate import evaluate_export_gate
from qa.translation_qa import summarize_flags
import main


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_clear_non_bubble_panel_mask_flags_accepts_derived_card_panel_mask():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "translated": "A sincronização foi concluída.",
                        "background_rgb": [253, 194, 150],
                        "bubble_mask_source": "derived_card_panel_mask",
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 2127,
                            "expanded_mask_pixels": 7897,
                            "evidence_score": 1.0,
                        },
                        "qa_flags": [
                            "safe_text_box_recomputed",
                            "bbox_fallback_bubble_mask",
                            "mask_outside_balloon_critical",
                        ],
                        "route_action": "review_required",
                        "route_reason": "mask_outside_balloon_critical",
                        "render_policy": "review_required",
                        "needs_review": True,
                    }
                ],
            }
        ]
    }

    cleared = main._clear_non_bubble_panel_mask_flags(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert cleared == 1
    assert layer["qa_flags"] == ["safe_text_box_recomputed"]
    assert layer["route_action"] == "translate_inpaint_render"
    assert "route_reason" not in layer
    assert "render_policy" not in layer
    assert "needs_review" not in layer


def test_render_plan_qa_flags_propagate_to_project_and_audit(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_003",
                "qa_flags": ["render_on_art_suspected", "ocr_run_on_suspect"],
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
                "numero": 1,
                "text_layers": [
                    {"id": "ocr_003", "translated": "Sombra", "qa_flags": []},
                    {"id": "ocr_004", "translated": "OK", "qa_flags": []},
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = summarize_flags(project["paginas"][0]["text_layers"])
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert "render_on_art_suspected" in layer["qa_flags"]
    assert "ocr_run_on_suspect" in layer["qa_flags"]
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []
    assert project["qa"]["summary"]["critical_count"] == project["qa"]["export_gate"]["critical_issue_count"]

    saved = json.loads(
        (
            work_dir
            / "debug"
            / "e2e"
            / "11_qa_export_gate"
            / "qa_flag_propagation_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["summary"]["qa_flag_not_propagated_count"] == 0


def test_review_route_preserves_existing_render_geometry_for_visual_fallback(tmp_path):
    work_dir = tmp_path
    decision_path = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_012" / "mask_decision.json"
    decision_path.parent.mkdir(parents=True)
    decision_path.write_text(
        json.dumps(
            {
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_002_band_012",
                "band_id": "page_002_band_012",
                "flags": ["source_glyph_area_ratio_critical", "weak_text_residual_after_inpaint"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "trace_id": "ocr_002@page_002_band_012",
                        "band_id": "page_002_band_012",
                        "translated": "ENTÃO, POR FAVOR, ESPERE UM POUCO MAIS.",
                        "safe_text_box": [132, 11293, 339, 11434],
                        "_debug_safe_text_box": [132, 11293, 339, 11434],
                        "render_bbox": [136, 11302, 335, 11425],
                        "qa_flags": ["weak_text_residual_after_inpaint"],
                    }
                ],
            }
        ],
        "qa": {},
    }

    main._propagate_debug_qa_flags_to_project(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert layer["route_action"] == "review_required"
    assert layer["render_policy"] == "review_required"
    assert layer["needs_review"] is True
    assert layer["safe_text_box"] == [132, 11293, 339, 11434]
    assert layer["render_bbox"] == [136, 11302, 335, 11425]


def test_merged_layer_source_trace_ids_satisfy_debug_flag_propagation(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_004",
                "trace_id": "ocr_004@page_003_band_048",
                "band_id": "page_003_band_048",
                "qa_flags": ["render_outside_balloon"],
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
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_048",
                        "band_id": "page_003_band_048",
                        "source_trace_ids": [
                            "ocr_001@page_003_band_048",
                            "ocr_004@page_003_band_048",
                        ],
                        "source_text_ids": ["ocr_001", "ocr_004"],
                        "merge_reason": "clustered_line_fragments",
                        "translated": "texto mesclado",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = summarize_flags(project["paginas"][0]["text_layers"])
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert "render_outside_balloon" in layer["qa_flags"]
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []
    assert project["qa"]["export_gate"]["status"] == "BLOCK"
    assert project["qa"]["export_gate"]["critical_issue_count"] == 1
    assert project["qa"]["export_gate"]["issues"][0]["type"] == "p0_render_blocker"


def test_merged_balloon_layer_uses_real_bubble_mask_for_render_safe_area():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_002_band_014",
                        "band_id": "page_002_band_014",
                        "translated": (
                            "Afinal, é câncer, por que se preocupar em usar um "
                            "empréstimo privado para um paciente? sua vida também é tão frustrante"
                        ),
                        "target_bbox": [8, 11794, 749, 12573],
                        "bbox": [301, 12418, 690, 12587],
                        "render_bbox": [349, 12439, 690, 12587],
                        "safe_text_box": [301, 12418, 690, 12587],
                        "_debug_safe_text_box": [301, 12418, 690, 12587],
                        "balloon_bbox": [8, 11794, 749, 12573],
                        "bubble_mask_bbox": [246, 12296, 746, 12670],
                        "bubble_inner_bbox": [59, 11845, 698, 12522],
                        "bubble_mask_source": "image_contour_bubble_mask",
                        "qa_flags": [
                            "same_balloon_fragment_merged",
                            "safe_text_box_recomputed",
                            "TEXT_OVERFLOW",
                        ],
                    }
                ],
            }
        ]
    }

    audit = main._repair_project_real_bubble_body_safe_areas(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert audit["safe_area_repaired_count"] == 1
    assert layer["layout_safe_reason"] == "merged_real_bubble_mask_bbox"
    assert layer["safe_text_box"][0] < 301
    assert layer["safe_text_box"][1] < 12418
    assert layer["safe_text_box"][2] > 690
    assert layer["render_bbox"] == layer["safe_text_box"]
    assert "TEXT_OVERFLOW" not in layer["qa_flags"]


def test_merged_balloon_layer_clamps_overbroad_safe_area_to_real_bubble_mask():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_002_band_005",
                        "band_id": "page_002_band_005",
                        "translated": "POR FAVOR, PELO BEM DA CRIANÇA.",
                        "target_bbox": [25, 4473, 667, 4745],
                        "bbox": [499, 4576, 656, 4661],
                        "render_bbox": [152, 4598, 540, 4620],
                        "safe_text_box": [102, 4506, 590, 4712],
                        "_debug_safe_text_box": [102, 4506, 590, 4712],
                        "balloon_bbox": [25, 4357, 667, 4629],
                        "bubble_mask_bbox": [435, 4516, 701, 4703],
                        "bubble_mask_source": "image_contour_bubble_mask",
                        "qa_flags": [
                            "same_balloon_fragment_merged",
                            "same_band_dependent_fragment_merged",
                            "mask_outside_balloon_critical",
                            "safe_text_box_recomputed",
                        ],
                    }
                ],
            }
        ]
    }

    audit = main._repair_project_real_bubble_body_safe_areas(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert audit["safe_area_repaired_count"] == 1
    assert layer["layout_safe_reason"] == "merged_real_bubble_mask_bbox"
    assert layer["safe_text_box"][0] >= 435
    assert layer["safe_text_box"][2] <= 701
    assert layer["render_bbox"] == layer["safe_text_box"]
    assert "mask_outside_balloon_critical" not in layer["qa_flags"]


def test_page_space_typeset_inputs_repair_overbroad_safe_area_before_render():
    page_texts = [
        {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_002_band_005",
            "band_id": "page_002_band_005",
            "translated": "POR FAVOR, PELO BEM DA CRIANÇA.",
            "target_bbox": [25, 4473, 667, 4745],
            "bbox": [499, 4576, 656, 4661],
            "render_bbox": [152, 4598, 540, 4620],
            "safe_text_box": [102, 4506, 590, 4712],
            "_debug_safe_text_box": [102, 4506, 590, 4712],
            "balloon_bbox": [25, 4357, 667, 4629],
            "bubble_mask_bbox": [435, 4516, 701, 4703],
            "bubble_mask_source": "image_contour_bubble_mask",
            "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
        }
    ]

    repaired, audit = main._repair_page_space_text_layers_for_typeset(page_texts, page_number=2)

    assert audit["safe_area_repaired_count"] == 1
    assert repaired[0]["safe_text_box"][0] >= 435
    assert repaired[0]["render_bbox"] == repaired[0]["safe_text_box"]


def test_page_space_typeset_inputs_scrub_stale_layout_bbox_before_render():
    page_texts = [
        {
            "id": "ocr_004",
            "text_id": "ocr_004",
            "trace_id": "ocr_004@page_002_band_007",
            "band_id": "page_002_band_007",
            "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O PRINCIPAL",
            "target_bbox": [461, 7044, 754, 7279],
            "bbox": [527, 7113, 688, 7198],
            "layout_bbox": [556, 678, 661, 800],
            "safe_text_box": [490, 7086, 725, 7237],
            "render_bbox": [490, 7086, 725, 7237],
            "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
        }
    ]

    repaired, audit = main._repair_page_space_text_layers_for_typeset(page_texts, page_number=2)

    assert audit["auxiliary_bbox_scrubbed_count"] == 1
    assert "layout_bbox" not in repaired[0]
    flags = main._page_text_coordinate_audit_flags(repaired, height=16383, width=800)
    assert "layout_bbox_coordinate_mismatch" not in flags


def test_render_plan_ignores_bbox_overreach_when_broad_bbox_does_not_drive_mask(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_001_band_002",
                "qa_flags": ["bbox_overreach"],
                "qa_metrics": {
                    "bbox_overreach": {
                        "ratio": 12.0,
                        "has_line_polygon_geometry": True,
                        "broad_bbox_drives_mask": False,
                    },
                    "render_balloon_containment": 1.0,
                },
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
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_003",
                        "trace_id": "ocr_003@page_001_band_002",
                        "translated": "Sombra",
                        "qa_flags": [],
                    },
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert "bbox_overreach" not in project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert audit["summary"]["render_plan_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_render_candidate_hydration_shifts_nested_qa_metric_bboxes(tmp_path):
    work_dir = tmp_path
    debug_root = work_dir / "debug" / "e2e"
    render_candidates = debug_root / "09_typeset" / "render_plan_candidates.jsonl"
    layout_blocks = debug_root / "05_layout_geometry" / "layout_blocks.jsonl"
    render_candidates.parent.mkdir(parents=True)
    layout_blocks.parent.mkdir(parents=True)
    render_candidates.write_text(
        json.dumps(
            {
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_025_band_067",
                "band_id": "page_025_band_067",
                "target_bbox": [463, 374, 629, 445],
                "safe_text_box": [466, 374, 625, 445],
                "render_bbox": [468, 380, 624, 438],
                "fit_status": "ok",
                "qa_metrics": {
                    "bbox_overreach": {
                        "bbox": [463, 374, 629, 445],
                        "text_geometry_bbox": [463, 428, 631, 446],
                        "broad_bbox_drives_mask": False,
                    },
                    "render_balloon_containment": 1.0,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    layout_blocks.write_text(
        json.dumps(
            {
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_025_band_067",
                "band_id": "page_025_band_067",
                "bboxes": {
                    "bbox": {"value": [463, 933, 629, 1004]},
                    "layout_bbox": {"value": [463, 933, 629, 1004]},
                },
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
                "numero": 23,
                "text_layers": [
                    {
                        "id": "ocr_003",
                        "trace_id": "ocr_003@page_025_band_067",
                        "band_id": "page_025_band_067",
                        "bbox": [463, 933, 629, 1004],
                        "translated": "Com medo sozinho?",
                        "qa_flags": [],
                    }
                ],
            }
        ],
    }

    audit = main._hydrate_project_render_metadata_from_debug_candidates(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert audit["hydrated_layers"] == 1
    assert layer["render_bbox"] == [468, 939, 624, 997]
    assert layer["safe_text_box"] == [466, 933, 625, 1004]
    overreach = layer["qa_metrics"]["bbox_overreach"]
    assert overreach["bbox"] == [463, 933, 629, 1004]
    assert overreach["text_geometry_bbox"] == [463, 987, 631, 1005]


def test_render_plan_ignores_stale_render_geometry_flags_when_final_render_is_contained(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_002_band_019",
                "band_id": "page_002_band_019",
                "qa_flags": ["TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"],
                "target_bbox": [297, 3535, 526, 3620],
                "safe_text_box": [328, 3544, 504, 3609],
                "render_bbox": [346, 3549, 485, 3604],
                "balloon_bbox": [298, 3523, 533, 3630],
                "qa_metrics": {"render_balloon_containment": 1.0},
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
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_002_band_019",
                        "band_id": "page_002_band_019",
                        "translated": "texto",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert project["paginas"][0]["text_layers"][0]["qa_flags"] == []
    assert audit["summary"]["render_plan_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_render_plan_propagates_render_fit_flags_even_when_top_level_flag_was_dropped(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_004_band_100",
                "band_id": "page_004_band_100",
                "qa_flags": ["safe_text_box_recomputed"],
                "target_bbox": [171, 10866, 262, 10905],
                "safe_text_box": [144, 10888, 535, 10919],
                "render_bbox": [173, 10900, 506, 10919],
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
                "numero": 5,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "trace_id": "ocr_001@page_004_band_100",
                        "translated": "texto",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert "TEXT_OVERFLOW" in project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert "safe_text_box_recomputed" in project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert audit["summary"]["render_plan_flags"] == 2
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_inpaint_residual_flags_propagate_to_project_and_export_gate(tmp_path):
    work_dir = tmp_path
    inpaint_decision = work_dir / "debug" / "e2e" / "08_inpaint" / "page_006_band_116" / "inpaint_decision.json"
    inpaint_decision.parent.mkdir(parents=True)
    inpaint_decision.write_text(
        json.dumps(
            {
                "text_ids": ["ocr_001"],
                "trace_ids": ["ocr_001@page_006_band_116"],
                "flags": ["text_residual_after_inpaint"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 6,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_006_band_116",
                        "page_id": "page_006",
                        "band_id": "page_006_band_116",
                        "translated": "texto",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = summarize_flags(project["paginas"][0]["text_layers"])
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert "text_residual_after_inpaint" in layer["qa_flags"]
    assert audit["summary"]["inpaint_decision_flags"] == 1
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert project["qa"]["export_gate"]["status"] == "BLOCK"


def test_inpaint_claim_for_merged_same_balloon_layers_propagates_to_primary_layer(tmp_path):
    work_dir = tmp_path
    inpaint_decision = work_dir / "debug" / "e2e" / "08_inpaint" / "page_002_band_005" / "inpaint_decision.json"
    inpaint_decision.parent.mkdir(parents=True)
    inpaint_decision.write_text(
        json.dumps(
            {
                "text_ids": ["ocr_001", "ocr_002"],
                "trace_ids": [
                    "ocr_001@page_002_band_005",
                    "ocr_002@page_002_band_005",
                ],
                "flags": [
                    "mask_outside_balloon_critical",
                    "weak_text_residual_after_inpaint",
                ],
            },
            ensure_ascii=False,
        ),
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
                        "trace_id": "ocr_001@page_002_band_005",
                        "source_trace_ids": [
                            "ocr_001@page_002_band_005",
                            "ocr_002@page_002_band_005",
                        ],
                        "band_id": "page_002_band_005",
                        "translated": "POR FAVOR",
                        "route_action": "merged_into_primary",
                        "qa_flags": ["debug_derived_bubble_mask_rejected"],
                    },
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "trace_id": "ocr_002@page_002_band_005",
                        "source_trace_ids": [
                            "ocr_001@page_002_band_005",
                            "ocr_002@page_002_band_005",
                        ],
                        "band_id": "page_002_band_005",
                        "translated": "POR FAVOR",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": ["same_balloon_fragment_merged"],
                    },
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    merged, primary = project["paginas"][0]["text_layers"]
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []
    assert "mask_outside_balloon_critical" not in merged["qa_flags"]
    assert "mask_outside_balloon_critical" in primary["qa_flags"]
    assert "weak_text_residual_after_inpaint" in primary["qa_flags"]
    assert primary["route_action"] == "review_required"
    assert primary["route_reason"] == "mask_outside_balloon_critical"


def test_visible_render_texts_suppresses_route_action_merged_into_primary():
    texts = [
        {
            "id": "ocr_003",
            "translated": "O DIRETOR",
            "visible": True,
            "route_action": "merged_into_primary",
            "render_policy": "normal",
        },
        {
            "id": "ocr_004",
            "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES",
            "route_action": "translate_inpaint_render",
            "render_policy": "normal",
        },
    ]

    visible = main._visible_render_texts(texts)

    assert [item["id"] for item in visible] == ["ocr_004"]


def test_page_has_final_renderable_text_ignores_merged_fragments_without_name_error():
    page = {
        "text_layers": [
            {
                "id": "fragment",
                "translated": "ESTOU MORRENDO DE FOME",
                "route_action": "merged_into_primary",
                "render_policy": "merged_into_primary",
            },
            {
                "id": "primary",
                "translated": "QUEM ESTA PAGANDO HOJE?",
                "route_action": "translate_inpaint_render",
                "render_policy": "normal",
            },
        ]
    }

    assert main._page_has_final_renderable_text(page) is True
    assert main._page_has_final_renderable_text({"text_layers": [page["text_layers"][0]]}) is False


def test_restore_missing_render_candidate_does_not_make_merged_fragment_visible():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "text": "HEY, LET'S GO! I'M STARVING",
                        "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                        "bbox": [148, 7192, 310, 7212],
                        "source_bbox": [148, 7192, 310, 7212],
                        "text_pixel_bbox": [148, 7192, 310, 7212],
                        "balloon_bbox": [133, 7129, 327, 7271],
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                        "visible": True,
                    }
                ],
            }
        ]
    }
    candidates = [
        {
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "render_bbox": [152, 7192, 304, 7217],
            "safe_text_box": [85, 7176, 393, 7228],
            "bbox": [148, 7192, 310, 7212],
        },
        {
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME",
            "route_action": "merged_into_primary",
            "render_policy": "merged_into_primary",
            "render_bbox": [270, 7629, 537, 7686],
            "safe_text_box": [177, 7603, 537, 7686],
            "bbox": [345, 7665, 537, 7686],
        },
    ]

    restored = main._restore_missing_render_candidate_layers(project, candidates)

    assert restored == 0
    assert len(project["paginas"][0]["text_layers"]) == 1


def test_restore_missing_render_candidate_does_not_restore_same_identity_fragment_when_primary_is_merged():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "text": "HEY, LET'S GO! I'M STARVING",
                        "translated": "EI, VAMOS!\nESTOU MORRENDO DE FOME",
                        "bbox": [148, 7248, 537, 7686],
                        "source_bbox": [29, 7109, 642, 7722],
                        "text_pixel_bbox": [148, 7248, 537, 7686],
                        "balloon_bbox": [133, 7185, 327, 7327],
                        "safe_text_box": [152, 7210, 308, 7302],
                        "render_bbox": [152, 7210, 308, 7302],
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                        "visible": True,
                        "qa_flags": ["same_balloon_fragment_merged"],
                    }
                ],
            }
        ]
    }
    candidates = [
        {
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "render_bbox": [152, 7248, 304, 7273],
            "safe_text_box": [85, 7232, 393, 7284],
            "bbox": [148, 7248, 310, 7268],
        },
        {
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME",
            "render_bbox": [270, 7629, 537, 7686],
            "safe_text_box": [177, 7603, 537, 7686],
            "bbox": [345, 7665, 537, 7686],
        },
    ]

    restored = main._restore_missing_render_candidate_layers(project, candidates)

    assert restored == 0
    assert len(project["paginas"][0]["text_layers"]) == 1


def test_suppress_same_identity_merged_fragments_hides_restored_visible_fragment():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "translated": "QUEM ESTÁ PAGANDO HOJE?\nESTOU MORRENDO DE FOME",
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                        "qa_flags": ["same_balloon_fragment_merged"],
                    },
                    {
                        "id": "ocr_001_fragment_2",
                        "text_id": "ocr_001_fragment_2",
                        "trace_id": "ocr_001@page_003_band_035#fragment_2",
                        "band_id": "page_003_band_035",
                        "translated": "ESTOU MORRENDO DE FOME",
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                        "qa_flags": ["fast_fill_no_glyph_evidence"],
                    },
                ],
            }
        ]
    }

    suppressed = main._suppress_same_identity_merged_fragments(project)

    fragment = project["paginas"][0]["text_layers"][1]
    assert suppressed == 1
    assert fragment["visible"] is False
    assert fragment["route_action"] == "merged_into_primary"
    assert fragment["render_policy"] == "merged_into_primary"


def test_hydration_reassigns_split_lobe_text_to_neighbor_balloon(tmp_path):
    debug_root = tmp_path / "debug" / "e2e" / "09_typeset"
    debug_root.mkdir(parents=True)
    rows = [
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "bbox": [148, 235, 310, 255],
            "render_bbox": [152, 235, 304, 260],
            "safe_text_box": [85, 219, 393, 271],
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
        },
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME",
            "bbox": [345, 652, 537, 673],
            "render_bbox": [270, 616, 537, 673],
            "safe_text_box": [177, 590, 537, 673],
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
        },
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "QUEM ESTA PAGANDO HOJE?",
            "bbox": [325, 634, 549, 751],
            "render_bbox": [361, 673, 488, 743],
            "safe_text_box": [343, 662, 506, 755],
            "source_trace_ids": ["ocr_003@page_003_band_035"],
            "source_text_ids": ["ocr_003"],
        },
    ]
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (debug_root / "render_plan_candidates.jsonl").write_text(payload, encoding="utf-8")
    (debug_root / "render_plan_raw.jsonl").write_text(payload, encoding="utf-8")
    project = {
        "_work_dir": str(tmp_path),
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "translated": "EI, VAMOS! ESTOU MORRENDO DE FOME",
                        "bbox": [148, 7192, 310, 7212],
                        "text_pixel_bbox": [148, 7192, 310, 7212],
                        "source_bbox": [148, 7192, 310, 7212],
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "translated": "QUEM ESTA PAGANDO HOJE?",
                        "bbox": [344, 7702, 540, 7761],
                        "text_pixel_bbox": [344, 7702, 540, 7761],
                        "source_bbox": [344, 7702, 540, 7761],
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                    },
                ],
            }
        ],
    }

    main._hydrate_project_render_metadata_from_debug_candidates(project)
    main._merge_same_balloon_fragment_layers(project)
    main._hydrate_project_render_metadata_from_debug_candidates(project)

    top, lower = project["paginas"][0]["text_layers"]
    assert top["translated"] == "EI, VAMOS!"
    assert top["render_bbox"] == [152, 235, 304, 260]
    assert lower["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?"
    assert lower["translated"].count("ESTOU MORRENDO DE FOME") == 1
    assert lower["source_trace_ids"] == [
        "ocr_001@page_003_band_035",
        "ocr_003@page_003_band_035",
    ]


def test_hydration_keeps_repaired_split_lobe_on_text_matching_page_space_candidate(tmp_path):
    debug_root = tmp_path / "debug" / "e2e" / "09_typeset"
    debug_root.mkdir(parents=True)
    rows = [
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "bbox": [148, 235, 310, 255],
            "render_bbox": [152, 235, 304, 260],
            "safe_text_box": [85, 219, 393, 271],
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
        },
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME",
            "bbox": [345, 652, 537, 673],
            "render_bbox": [270, 616, 537, 673],
            "safe_text_box": [177, 590, 537, 673],
            "source_trace_ids": ["ocr_001@page_003_band_035"],
            "source_text_ids": ["ocr_001"],
        },
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "QUEM ESTA PAGANDO HOJE?",
            "bbox": [325, 634, 549, 751],
            "render_bbox": [361, 673, 488, 743],
            "safe_text_box": [343, 662, 506, 755],
            "source_trace_ids": ["ocr_003@page_003_band_035"],
            "source_text_ids": ["ocr_003"],
        },
        {
            "candidate_kind": "layout_fit",
            "text_id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?",
            "bbox": [325, 634, 549, 751],
            "render_bbox": [270, 616, 537, 673],
            "safe_text_box": [177, 590, 537, 673],
            "source_trace_ids": [
                "ocr_001@page_003_band_035",
                "ocr_003@page_003_band_035",
            ],
            "source_text_ids": ["ocr_001", "ocr_003"],
            "qa_flags": ["same_balloon_fragment_merged"],
        },
    ]
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (debug_root / "render_plan_candidates.jsonl").write_text(payload, encoding="utf-8")
    (debug_root / "render_plan_raw.jsonl").write_text(payload, encoding="utf-8")
    project = {
        "_work_dir": str(tmp_path),
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "translated": "EI, VAMOS!",
                        "bbox": [345, 7567, 537, 7588],
                        "text_pixel_bbox": [345, 7567, 537, 7588],
                        "source_bbox": [345, 7567, 537, 7588],
                        "render_bbox": [270, 7531, 537, 7588],
                        "safe_text_box": [177, 7505, 537, 7588],
                        "balloon_bbox": [265, 7470, 629, 7741],
                        "bubble_mask_bbox": [265, 7470, 629, 7741],
                        "bubble_inner_bbox": [301, 7518, 593, 7693],
                        "coordinate_space": "page",
                        "source_coordinate_space": "page",
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_003_band_035",
                        "band_id": "page_003_band_035",
                        "translated": "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?",
                        "bbox": [325, 7549, 549, 7666],
                        "text_pixel_bbox": [325, 7549, 549, 7666],
                        "source_bbox": [325, 7549, 549, 7666],
                        "render_bbox": [361, 7588, 488, 7658],
                        "safe_text_box": [343, 7577, 506, 7670],
                        "coordinate_space": "page",
                        "source_coordinate_space": "page",
                        "visible": True,
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                    },
                ],
            }
        ],
    }

    main._hydrate_project_render_metadata_from_debug_candidates(project)

    top, lower = project["paginas"][0]["text_layers"]
    assert top["translated"] == "EI, VAMOS!"
    assert top["bbox"] == [148, 7150, 310, 7170]
    assert top["render_bbox"] == [152, 7150, 304, 7175]
    assert top["safe_text_box"] == [85, 7134, 393, 7186]
    assert "_same_band_restore_coordinate_offset" not in top
    assert lower["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?"

    main._merge_same_balloon_fragment_layers(project)
    main._normalize_final_project_page_space_layers(project)
    main._hydrate_project_render_metadata_from_debug_candidates(project)
    main._merge_same_balloon_fragment_layers(project)
    main._normalize_final_project_page_space_layers(project)

    visible_layers = [
        layer
        for layer in project["paginas"][0]["text_layers"]
        if layer.get("visible", True) is not False
    ]
    assert [layer["id"] for layer in visible_layers] == ["ocr_001", "ocr_003"]
    assert visible_layers[0]["translated"] == "EI, VAMOS!"
    assert visible_layers[0]["render_bbox"] == [152, 7150, 304, 7175]
    assert visible_layers[1]["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTA PAGANDO HOJE?"
    assert visible_layers[1]["render_bbox"][1] > visible_layers[0]["render_bbox"][3] + 250
    assert visible_layers[1]["safe_text_box"][1] > visible_layers[0]["safe_text_box"][3] + 250


def test_split_lobe_repair_restores_owner_to_first_direct_candidate_when_payload_was_contaminated():
    project_layers = [
        {
            "id": "ocr_001",
            "text_id": "ocr_001",
            "trace_id": "ocr_001@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "QUEM ESTÁ PAGANDO HOJE?\nESTOU MORRENDO DE FOME",
            "source_trace_ids": [
                "ocr_003@page_003_band_035",
                "ocr_001@page_003_band_035",
            ],
            "bbox": [148, 7248, 537, 7686],
            "visible": True,
            "route_action": "translate_inpaint_render",
            "render_policy": "normal",
        },
        {
            "id": "ocr_003",
            "text_id": "ocr_003",
            "trace_id": "ocr_003@page_003_band_035",
            "band_id": "page_003_band_035",
            "translated": "QUEM ESTÁ PAGANDO HOJE?",
            "bbox": [344, 7702, 540, 7761],
            "visible": True,
            "route_action": "translate_inpaint_render",
            "render_policy": "normal",
        },
    ]
    candidates = [
        {
            "trace_id": "ocr_001@page_003_band_035",
            "text_id": "ocr_001",
            "band_id": "page_003_band_035",
            "translated": "EI, VAMOS!",
            "bbox": [148, 7248, 310, 7268],
            "render_bbox": [152, 7248, 304, 7273],
            "safe_text_box": [85, 7232, 393, 7284],
        },
        {
            "trace_id": "ocr_001@page_003_band_035",
            "text_id": "ocr_001",
            "band_id": "page_003_band_035",
            "translated": "ESTOU MORRENDO DE FOME",
            "bbox": [345, 7665, 537, 7686],
            "render_bbox": [270, 7629, 537, 7686],
            "safe_text_box": [177, 7603, 537, 7686],
        },
        {
            "trace_id": "ocr_003@page_003_band_035",
            "text_id": "ocr_003",
            "band_id": "page_003_band_035",
            "translated": "QUEM ESTÁ PAGANDO HOJE?",
            "bbox": [276, 7612, 598, 7799],
            "render_bbox": [361, 7686, 488, 7756],
            "safe_text_box": [343, 7675, 506, 7768],
        },
    ]

    repaired = main._repair_project_split_lobe_text_payloads(project_layers, candidates)

    assert repaired >= 2
    assert project_layers[0]["translated"] == "EI, VAMOS!"
    assert project_layers[1]["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?"
    assert project_layers[1]["source_trace_ids"] == [
        "ocr_001@page_003_band_035",
        "ocr_003@page_003_band_035",
    ]


def test_debug_mask_flags_are_cleared_when_final_mask_decision_has_no_claim(tmp_path):
    work_dir = tmp_path
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_001",
                        "translated": "texto",
                        "qa_flags": ["mask_outside_balloon_critical", "ocr_run_on_suspect"],
                    }
                ],
            }
        ],
        "qa": {},
    }

    main._propagate_debug_qa_flags_to_project(project)

    flags = project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert "mask_outside_balloon_critical" not in flags
    assert "ocr_run_on_suspect" in flags


def test_render_plan_mask_flags_do_not_repropagate_without_mask_decision(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_001_band_001",
                "qa_flags": ["mask_outside_balloon_critical", "ocr_run_on_suspect"],
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
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_001",
                        "translated": "texto",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    main._propagate_debug_qa_flags_to_project(project)

    flags = project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert "mask_outside_balloon_critical" not in flags
    assert "ocr_run_on_suspect" in flags


def test_mask_flags_use_band_identity_when_text_ids_repeat(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_007" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_007",
                "text_id": "ocr_001",
                "text_ids": ["ocr_001"],
                "flags": ["mask_density_high"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_001",
                        "text_instance_id": "page_001_band_001_ocr_001",
                        "band_id": "page_001_band_001",
                        "qa_flags": [],
                    }
                ],
            },
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_002_band_007",
                        "text_instance_id": "page_002_band_007_ocr_001",
                        "band_id": "page_002_band_007",
                        "qa_flags": [],
                    }
                ],
            },
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    first_layer = project["paginas"][0]["text_layers"][0]
    second_layer = project["paginas"][1]["text_layers"][0]
    assert "mask_density_high" not in first_layer["qa_flags"]
    assert "mask_density_high" not in second_layer["qa_flags"]
    assert audit["summary"]["mask_decision_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_new_bubble_mask_contract_flags_propagate_from_mask_decision(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_007" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_007",
                "text_id": "ocr_001",
                "trace_ids": ["ocr_001@page_002_band_007"],
                "flags": ["glyph_mask_outside_bubble", "bbox_fallback_bubble_mask"],
            },
            ensure_ascii=False,
        ),
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
                        "trace_id": "ocr_001@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    flags = project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert "glyph_mask_outside_bubble" in flags
    assert "bbox_fallback_bubble_mask" in flags
    assert audit["summary"]["mask_decision_flags"] == 2


def test_clean_inpaint_decision_suppresses_mask_geometry_flags(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_018" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_018",
                "text_id": "ocr_001",
                "trace_ids": ["ocr_001@page_002_band_018"],
                "flags": ["mask_density_high", "mask_outside_balloon_critical"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inpaint_decision = work_dir / "debug" / "e2e" / "08_inpaint" / "page_002_band_018" / "inpaint_decision.json"
    inpaint_decision.parent.mkdir(parents=True)
    inpaint_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_018",
                "trace_ids": ["ocr_001@page_002_band_018"],
                "flags": [],
                "changed_pixels_outside_effective_limit": 0,
                "cleanup_changed_outside_limit_mask": 0,
                "residual_text": {"has_residual": False},
            },
            ensure_ascii=False,
        ),
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
                        "trace_id": "ocr_001@page_002_band_018",
                        "band_id": "page_002_band_018",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert project["paginas"][0]["text_layers"][0]["qa_flags"] == []
    assert audit["summary"]["mask_decision_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_clean_inpaint_decision_allows_tiny_cleanup_limit_noise(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_019" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_019",
                "text_ids": ["ocr_001", "ocr_002"],
                "trace_ids": ["ocr_001@page_002_band_019", "ocr_002@page_002_band_019"],
                "flags": ["mask_outside_balloon_critical"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inpaint_decision = work_dir / "debug" / "e2e" / "08_inpaint" / "page_002_band_019" / "inpaint_decision.json"
    inpaint_decision.parent.mkdir(parents=True)
    inpaint_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_019",
                "trace_ids": ["ocr_001@page_002_band_019", "ocr_002@page_002_band_019"],
                "flags": [],
                "changed_pixels_outside_effective_limit": 0,
                "cleanup_changed_outside_limit_mask": 4,
                "residual_text": {"has_residual": False},
            },
            ensure_ascii=False,
        ),
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
                        "trace_id": "ocr_001@page_002_band_019",
                        "source_trace_ids": ["ocr_001@page_002_band_019", "ocr_002@page_002_band_019"],
                        "band_id": "page_002_band_019",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert project["paginas"][0]["text_layers"][0]["qa_flags"] == []
    assert audit["summary"]["mask_decision_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_low_inpaint_decision_flag_without_project_layer_does_not_block_traceability(tmp_path):
    work_dir = tmp_path
    inpaint_decision = work_dir / "debug" / "e2e" / "08_inpaint" / "page_003_band_048" / "inpaint_decision.json"
    inpaint_decision.parent.mkdir(parents=True)
    inpaint_decision.write_text(
        json.dumps(
            {
                "band_id": "page_003_band_048",
                "trace_ids": ["ocr_001@page_003_band_048"],
                "flags": ["fast_fill_without_raw_mask"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 3, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert audit["summary"]["inpaint_decision_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_unmatched_mask_decision_claim_blocks_as_traceability_failure(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_003_band_042" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_003_band_042",
                "trace_ids": ["ocr_999@page_003_band_042"],
                "flags": ["mask_outside_balloon_critical"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_042",
                        "band_id": "page_003_band_042",
                        "translated": "texto",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags(project["paginas"][0]["text_layers"]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["mask_decision_flags"] == 1
    assert audit["summary"]["qa_flag_not_propagated_count"] == 1
    assert project["qa"]["summary"]["highest_severity"] == "critical"
    assert project["qa"]["summary"]["critical_count"] == 1
    assert project["qa"]["export_gate"]["status"] == "BLOCK"
    assert project["qa"]["export_gate"]["critical_issue_count"] == 1
    issue = project["qa"]["export_gate"]["issues"][0]
    assert issue["type"] == "p0_traceability_blocker"
    assert issue["missing_flag"] == "mask_outside_balloon_critical"


def test_unmatched_render_plan_review_flags_do_not_block_traceability(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_999",
                "trace_id": "ocr_999@page_003_band_053",
                "band_id": "page_003_band_053",
                "qa_flags": ["safe_text_box_recomputed", "ocr_partial_low_confidence_fragment"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 3, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags([]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []
    assert project["qa"]["export_gate"]["status"] == "PASS"


def test_negative_dark_claim_satisfied_by_same_band_project_layer_flag(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "negative_dark_000",
                "trace_id": "negative_dark_000@page_002_band_023",
                "band_id": "page_002_band_023",
                "qa_flags": ["render_on_art_suspected"],
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
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_002_band_023",
                        "band_id": "page_002_band_023",
                        "translated": "texto",
                        "qa_flags": ["render_on_art_suspected"],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []


def test_unmatched_fast_fill_no_glyph_evidence_debug_claims_are_review_only(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
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
    inpaint_root = work_dir / "debug" / "e2e" / "08_inpaint" / "page_051_band_127"
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
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 51, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags([]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["qa_flag_not_propagated_count"] == 2
    assert all(item["is_review_only"] for item in audit["missing_in_project"])
    assert project["qa"]["export_gate"]["status"] == "PASS"
    assert project["qa"]["export_gate"]["critical_issue_count"] == 0
    assert project["qa"]["export_gate"]["review_issue_count"] == 2
    assert all(issue["type"] == "needs_review" for issue in project["qa"]["export_gate"]["issues"])
    assert all(issue["flags"] == ["qa_flag_not_propagated"] for issue in project["qa"]["export_gate"]["issues"])


def test_unmatched_scanlation_watermark_render_fit_does_not_block_traceability(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_010_band_142",
                "band_id": "page_010_band_142",
                "original": "NEWTOKLAG",
                "translated": "NEWTOKLAG",
                "qa_flags": ["compact_small_text_capacity", "fit_below_minimum_legible"],
                "fit_status": "below_minimum_legible",
                "safe_text_box": [298, 4856, 438, 4873],
                "render_bbox": [298, 4862, 332, 4867],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 10, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags([]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["qa_flag_not_propagated_count"] == 0
    assert audit["missing_in_project"] == []
    assert project["qa"]["export_gate"]["status"] == "PASS"


def test_translator_note_missing_render_geometry_gets_best_effort_box():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_002_band_017",
                        "route_action": "translate_inpaint_render",
                        "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE DA MÁFIA.",
                        "balloon_bbox": [535, 14278, 797, 14413],
                        "qa_flags": ["safe_text_box_recomputed", "fit_below_minimum_legible"],
                        "fit_status": "below_minimum_legible",
                        "style": {"tamanho": 48, "force_upper": True},
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._ensure_project_render_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert audit["filled_fit_metadata_count"] == 1
    assert layer["safe_text_box"] == [535, 14278, 797, 14413]
    assert layer["render_bbox"] == [535, 14278, 797, 14413]
    assert layer["fit_status"] == "ok"
    assert "fit_below_minimum_legible" not in layer["qa_flags"]
    assert "translator_note_best_effort_render" in layer["qa_flags"]
    assert layer["style"]["tamanho"] <= 16
    assert layer["style"]["force_upper"] is False
    assert main._drop_stale_final_render_geometry(layer)["render_bbox"] == [535, 14278, 797, 14413]


def test_translator_note_existing_low_fit_geometry_gets_compact_best_effort_style():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_002_band_017",
                        "route_action": "translate_inpaint_render",
                        "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE DA MÁFIA.",
                        "safe_text_box": [567, 14298, 765, 14393],
                        "_debug_safe_text_box": [567, 14298, 765, 14393],
                        "target_bbox": [535, 14278, 797, 14413],
                        "render_bbox": [571, 14304, 761, 14387],
                        "qa_flags": ["safe_text_box_recomputed"],
                        "fit_status": "below_minimum_legible",
                        "style": {"tamanho": 48, "force_upper": True},
                        "estilo": {"tamanho": 48, "force_upper": True},
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._ensure_project_render_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert audit["filled_fit_metadata_count"] == 1
    assert layer["safe_text_box"] == [535, 14278, 797, 14413]
    assert layer["render_bbox"] == [535, 14278, 797, 14413]
    assert layer["fit_status"] == "ok"
    assert "translator_note_best_effort_render" in layer["qa_flags"]
    assert layer["style"]["tamanho"] <= 8
    assert layer["style"]["force_upper"] is False
    assert layer["_render_bbox_from_repaired_safe_text_box"] is True


def test_post_rerender_contract_repair_rerenders_when_translator_note_was_narrowed(tmp_path, monkeypatch):
    calls = []

    def fake_render_page_image(project, page_idx, out_img):
        calls.append((page_idx, str(out_img)))

    monkeypatch.setattr(main, "render_page_image", fake_render_page_image)
    project = {
        "_work_dir": str(tmp_path),
        "paginas": [
            {
                "numero": 2,
                "arquivo_original": "002.jpg",
                "arquivo_traduzido": "translated/002.jpg",
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_002_band_017",
                        "text": "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S MOB BOSS",
                        "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE DA MÁFIA",
                        "route_action": "translate_inpaint_render",
                        "render_policy": "normal",
                        "safe_text_box": [570, 14294, 762, 14397],
                        "render_bbox": [573, 14305, 759, 14386],
                        "target_bbox": [559, 14287, 773, 14404],
                        "balloon_bbox": [535, 14278, 797, 14413],
                        "style": {"fonte": "ComicNeue-Bold.ttf", "tamanho": 8},
                        "qa_flags": ["translator_note_best_effort_render"],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._rerender_final_project_images_after_contract(project, tmp_path)

    layer = project["paginas"][0]["text_layers"][0]
    assert layer["safe_text_box"] == [535, 14278, 797, 14413]
    assert layer["render_bbox"] == [535, 14278, 797, 14413]
    assert audit["post_rerender_contract_audit"]["filled_fit_metadata_count"] == 1
    assert audit["pages_rerendered"] == 1
    assert calls == [(0, str(tmp_path / "translated" / "002.jpg"))]


def test_unmatched_accepted_detect_candidate_blocks_export(tmp_path):
    work_dir = tmp_path
    detect_candidates = work_dir / "debug" / "e2e" / "02_strip_detect" / "detect_candidates.jsonl"
    detect_candidates.parent.mkdir(parents=True)
    detect_candidates.write_text(
        json.dumps(
            {
                "candidate_id": "page_003_band_030_cand_000",
                "page_id": "page_003",
                "band_id": "page_003_band_030",
                "bbox_page": [118, 7680, 223, 7766],
                "accepted": True,
                "matched_text_id": None,
                "matched_text_ids": [],
                "matched_trace_ids": [],
                "match_count": 0,
                "match_reason": "no_text_in_band",
                "match_method": "no_text_in_band",
                "band_text_count": 0,
                "has_inner_dark_text": True,
                "inner_dark_component_count": 7,
                "inner_dark_area": 925,
                "significant_component_count": 4,
                "significant_area": 800,
                "bright_pixel_ratio": 0.3297,
                "dark_pixel_ratio": 0.0967,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 3, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags([]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["detect_accepted_unmatched_count"] == 1
    assert project["qa"]["summary"]["highest_severity"] == "critical"
    assert project["qa"]["export_gate"]["status"] == "BLOCK"
    issue = project["qa"]["export_gate"]["issues"][0]
    assert issue["type"] == "p0_traceability_blocker"
    assert issue["flags"] == ["detect_candidate_without_ocr_text"]
    assert issue["candidate_id"] == "page_003_band_030_cand_000"


def test_unmatched_detect_candidates_require_strong_evidence_and_dedupe(tmp_path):
    work_dir = tmp_path
    detect_candidates = work_dir / "debug" / "e2e" / "02_strip_detect" / "detect_candidates.jsonl"
    detect_candidates.parent.mkdir(parents=True)
    rows = [
        {
            "run_id": "run-a",
            "candidate_id": "page_003_band_030_cand_000",
            "page_id": "page_003",
            "band_id": "page_003_band_030",
            "bbox_page": [118, 7680, 223, 7766],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 7,
            "inner_dark_area": 925,
            "significant_component_count": 4,
            "significant_area": 800,
            "bright_pixel_ratio": 0.3297,
            "dark_pixel_ratio": 0.0967,
        },
        {
            "run_id": "run-b",
            "candidate_id": "page_003_band_030_cand_000",
            "page_id": "page_003",
            "band_id": "page_003_band_030",
            "bbox_page": [118, 7680, 223, 7766],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 7,
            "inner_dark_area": 925,
            "significant_component_count": 4,
            "significant_area": 800,
            "bright_pixel_ratio": 0.3297,
            "dark_pixel_ratio": 0.0967,
        },
        {
            "run_id": "run-b",
            "candidate_id": "page_003_band_030_cand_000",
            "page_id": "page_003",
            "band_id": "page_003_band_030",
            "bbox_page": [118, 7680, 223, 7766],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 4,
            "inner_dark_area": 174,
            "significant_component_count": 0,
            "significant_area": 0,
            "bright_pixel_ratio": 0.30,
            "dark_pixel_ratio": 0.08,
        },
        {
            "run_id": "run-b",
            "candidate_id": "page_006_band_058_cand_000",
            "page_id": "page_006",
            "band_id": "page_006_band_058",
            "bbox_page": [296, 318, 416, 525],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 4,
            "inner_dark_area": 510,
            "significant_component_count": 2,
            "significant_area": 450,
            "bright_pixel_ratio": 0.3055,
            "dark_pixel_ratio": 0.0552,
        },
        {
            "run_id": "run-b",
            "candidate_id": "page_004_band_094_cand_000",
            "page_id": "page_004",
            "band_id": "page_004_band_094",
            "bbox_page": [204, 25584, 541, 25944],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 26,
            "inner_dark_area": 8025,
            "significant_component_count": 6,
            "significant_area": 7449,
            "bright_pixel_ratio": 0.20,
            "dark_pixel_ratio": 0.28,
        },
        {
            "run_id": "run-b",
            "candidate_id": "page_005_band_125_cand_000",
            "page_id": "page_005",
            "band_id": "page_005_band_125",
            "bbox_page": [86, 20822, 555, 21431],
            "accepted": True,
            "match_count": 0,
            "match_reason": "no_text_in_band",
            "match_method": "no_text_in_band",
            "has_inner_dark_text": True,
            "inner_dark_component_count": 73,
            "inner_dark_area": 10117,
            "significant_component_count": 9,
            "significant_area": 7505,
            "bright_pixel_ratio": 0.4884,
            "dark_pixel_ratio": 0.0396,
        },
    ]
    detect_candidates.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [{"numero": 3, "text_layers": []}],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    project["qa"]["summary"] = main._augment_qa_summary_with_debug_contract(
        summarize_flags([]),
        audit,
    )
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    assert audit["summary"]["detect_accepted_unmatched_count"] == 1
    assert [item["candidate_id"] for item in audit["unmatched_detect_candidates"]] == [
        "page_003_band_030_cand_000"
    ]
    assert project["qa"]["export_gate"]["status"] == "BLOCK"
    assert project["qa"]["export_gate"]["critical_issue_count"] == 1


def test_mask_decision_does_not_propagate_bbox_overreach_group_flags(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_002_band_018" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_002_band_018",
                "text_ids": ["ocr_002", "ocr_005"],
                "trace_ids": ["ocr_002@page_002_band_018", "ocr_005@page_002_band_018"],
                "flags": ["bbox_overreach"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "trace_id": "ocr_002@page_002_band_018",
                        "band_id": "page_002_band_018",
                        "qa_flags": [],
                    },
                    {
                        "id": "ocr_005",
                        "text_id": "ocr_005",
                        "trace_id": "ocr_005@page_002_band_018",
                        "band_id": "page_002_band_018",
                        "qa_flags": [],
                    },
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    assert project["paginas"][0]["text_layers"][0]["qa_flags"] == []
    assert project["paginas"][0]["text_layers"][1]["qa_flags"] == []
    assert audit["summary"]["mask_decision_flags"] == 0
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_mask_decision_propagates_source_glyph_area_ratio_critical(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_003_band_023" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_003_band_023",
                "text_id": "ocr_001",
                "text_ids": ["ocr_001"],
                "trace_ids": ["ocr_001@page_003_band_023"],
                "flags": ["source_glyph_area_ratio_critical"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    project = {
        "_work_dir": str(work_dir),
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_023",
                        "band_id": "page_003_band_023",
                        "route_action": "translate_inpaint_render",
                        "route_reason": "dialogue_balloon_with_english_text",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    main._ensure_project_route_action_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert "source_glyph_area_ratio_critical" in layer["qa_flags"]
    assert layer["route_action"] == "translate_inpaint_render"
    assert layer["route_reason"] == "dialogue_balloon_with_english_text"
    assert layer["render_policy"] == "normal"
    assert layer.get("needs_review") is not True
    assert audit["summary"]["mask_decision_flags"] == 1
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_mask_decision_source_glyph_critical_with_art_flag_routes_review(tmp_path):
    work_dir = tmp_path
    mask_decision = work_dir / "debug" / "e2e" / "06_mask_segmentation" / "page_003_band_023" / "mask_decision.json"
    mask_decision.parent.mkdir(parents=True)
    mask_decision.write_text(
        json.dumps(
            {
                "band_id": "page_003_band_023",
                "text_id": "ocr_001",
                "text_ids": ["ocr_001"],
                "trace_ids": ["ocr_001@page_003_band_023"],
                "flags": ["source_glyph_area_ratio_critical"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "trace_id": "ocr_001@page_003_band_023",
                "text_id": "ocr_001",
                "qa_flags": ["render_on_art_suspected"],
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
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_003_band_023",
                        "band_id": "page_003_band_023",
                        "route_action": "translate_inpaint_render",
                        "route_reason": "dialogue_balloon_with_english_text",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)
    main._ensure_project_route_action_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert {"source_glyph_area_ratio_critical", "render_on_art_suspected"}.issubset(set(layer["qa_flags"]))
    assert layer["route_action"] == "review_required"
    assert layer["route_reason"] == "source_glyph_area_ratio_critical"
    assert layer["render_policy"] == "review_required"
    assert layer["needs_review"] is True
    assert audit["summary"]["mask_decision_flags"] == 1


def test_qa_flag_identity_matching_falls_back_to_text_instance_id(tmp_path):
    work_dir = tmp_path
    render_plan = work_dir / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl"
    render_plan.parent.mkdir(parents=True)
    render_plan.write_text(
        json.dumps(
            {
                "trace_id": "stale-trace@page_001_band_005",
                "text_instance_id": "page_001_band_005_ocr_001",
                "text_id": "ocr_001",
                "qa_flags": ["render_on_art_suspected"],
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
                "numero": 1,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_001_band_005",
                        "text_instance_id": "page_001_band_005_ocr_001",
                        "band_id": "page_001_band_005",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "qa": {},
    }

    audit = main._propagate_debug_qa_flags_to_project(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert "render_on_art_suspected" in layer["qa_flags"]
    assert audit["summary"]["qa_flag_not_propagated_count"] == 0


def test_debug_export_gate_artifacts_link_all_critical_issues(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    project = {
        "qa": {
            "summary": {"critical_count": 2, "highest_severity": "critical"},
            "export_gate": {
                "status": "BLOCK",
                "critical_issue_count": 2,
                "review_issue_count": 0,
                "issues": [
                    {
                        "page": 1,
                        "layer": "ocr_003",
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": ["render_on_art_suspected"],
                    },
                    {
                        "page": 1,
                        "layer": "ocr_004",
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": ["bbox_overreach_critical"],
                    },
                ],
            },
        }
    }

    main._write_debug_export_gate_artifacts(recorder, project)

    gate_dir = tmp_path / "debug" / "e2e" / "11_qa_export_gate"
    visual_blockers = _read_jsonl(gate_dir / "visual_blockers.jsonl")
    qa_issues = _read_jsonl(gate_dir / "qa_issues.jsonl")

    assert [issue["layer"] for issue in visual_blockers] == ["ocr_003", "ocr_004"]
    assert all(issue["severity"] == "critical" for issue in visual_blockers)
    assert len(qa_issues) == 2
    assert all("09_typeset/render_plan_final.jsonl" in issue["linked_artifacts"] for issue in qa_issues)
    assert all("05_layout_geometry/layout_blocks.jsonl" in issue["linked_artifacts"] for issue in qa_issues)


def test_export_gate_debug_artifacts_keep_traceable_text_identity(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    project = {
        "idioma_origem": "en",
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_002_band_007",
                        "text_instance_id": "page_002_band_007_ocr_003",
                        "page_id": "page_002",
                        "band_id": "page_002_band_007",
                        "translated": "Sombra",
                        "qa_flags": ["render_on_art_suspected"],
                        "bbox": [10, 20, 80, 60],
                        "balloon_bbox": [5, 10, 100, 90],
                        "render_bbox": [140, 25, 200, 55],
                    }
                ],
            }
        ],
        "qa": {},
    }
    project["qa"]["summary"] = summarize_flags(project["paginas"][0]["text_layers"])
    project["qa"]["export_gate"] = evaluate_export_gate(project)

    main._write_debug_export_gate_artifacts(recorder, project)

    gate_dir = tmp_path / "debug" / "e2e" / "11_qa_export_gate"
    issue = _read_jsonl(gate_dir / "qa_issues.jsonl")[0]
    blocker = _read_jsonl(gate_dir / "visual_blockers.jsonl")[0]

    for payload in (issue, blocker):
        assert payload["trace_id"] == "ocr_003@page_002_band_007"
        assert payload["text_instance_id"] == "page_002_band_007_ocr_003"
        assert payload["text_id"] == "ocr_003"
        assert payload["page_id"] == "page_002"
        assert payload["band_id"] == "page_002_band_007"
        assert payload["render_bbox"] == [140, 25, 200, 55]
        assert payload["balloon_bbox"] == [5, 10, 100, 90]


def test_final_project_coordinate_audit_marks_mixed_page_texts():
    project = {
        "paginas": [
            {
                "numero": 1,
                "height": 13832,
                "width": 800,
                "text_layers": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "band_y_top": 5420,
                        "band_height": 895,
                        "bbox": [25, 5436, 667, 5745],
                        "balloon_bbox": [466, 5606, 696, 5777],
                        "bubble_inner_bbox": [513, 230, 649, 313],
                        "safe_text_box": [525, 242, 637, 301],
                        "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
                    }
                ],
            }
        ],
        "qa": {"summary": {}},
    }

    audit = main._apply_final_project_coordinate_audit(project)

    flags = project["paginas"][0]["text_layers"][0]["qa_flags"]
    assert audit["applied"] is True
    assert audit["flags_added"] > 0
    assert audit["flagged_pages"] == 1
    assert "page_space_rerender_mixed_coordinates" in flags
    assert "layout_bbox_coordinate_mismatch" in flags
