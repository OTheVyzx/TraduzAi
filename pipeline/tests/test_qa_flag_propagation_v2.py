import json
from pathlib import Path

from debug_tools import DebugRecorder
from qa.export_gate import evaluate_export_gate
from qa.translation_qa import summarize_flags
import main


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
                "significant_component_count": 3,
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
            "significant_component_count": 3,
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
            "significant_component_count": 3,
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
                        "render_bbox": [15, 25, 75, 55],
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
        assert payload["render_bbox"] == [15, 25, 75, 55]
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
