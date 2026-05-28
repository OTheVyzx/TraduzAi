import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "tools" / "analyze_e2e_debug.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_e2e_debug", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Pre-existing tests (kept; ensure backward compatibility).
# ---------------------------------------------------------------------------


def test_missing_artifacts_still_write_report(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    root.mkdir(parents=True)

    report = module.analyze_root(root, write_report=True)

    assert report["run_count"] == 0
    assert (root / "debug_report.json").exists()
    assert (root / "debug_report.md").exists()
    assert json.loads((root / "debug_report.json").read_text(encoding="utf-8"))[
        "runs"
    ] == []


def test_synthetic_run_reports_debug_guide_metrics(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "B_skip_inpaint"
    run.mkdir(parents=True)
    mojibake_text = "VOC" + "Ã" + "Š" + " SABE"

    _write_json(
        root / "run_status.json",
        [{"id": "B_skip_inpaint", "exit_code": 0}],
    )
    _write_json(
        run / "runner_config.json",
        {
            "skip_inpaint": True,
            "env": {
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
                "OTHER": "ignored",
            },
        },
    )
    _write_json(
        run / "debug" / "e2e" / "00_run" / "env_snapshot.json",
        {
            "env_vars": {
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1",
                "OTHER": "ignored",
            }
        },
    )
    _write_json(
        run / "qa_report.json",
        {
            "needs_review": True,
            "summary": {
                "flag_counts": {
                    "render_outside_balloon": 2,
                },
            },
        },
    )
    _write_json(
        run / "debug" / "e2e" / "11_qa_export_gate" / "export_gate.json",
        {
            "status": "BLOCK",
            "critical_issue_count": 3,
            "needs_review": True,
            "issues": [
                {"flags": ["render_on_art_suspected"]},
                {"flags": ["bbox_overreach_critical"]},
                {"flags": ["bbox_overreach_critical"]},
                {"flags": ["bbox_overreach_critical"]},
            ],
        },
    )
    _write_json(
        run
        / "debug"
        / "e2e"
        / "11_qa_export_gate"
        / "qa_export_gate_consistency.json",
        {
            "summary": {"critical_count": 0},
            "export_gate": {"status": "BLOCK", "critical_issue_count": 3},
            "consistent": False,
        },
    )
    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "texto_traduzido": mojibake_text,
                            "confianca_ocr": 0.0,
                            "bbox": [10, 20, 40, 20],
                            "source_bbox": [10, 520, 100, 80],
                            "balloon_bbox": [10, 520, 100, 80],
                            "content_class": "dialogue",
                        },
                        {
                            "id": "b",
                            "texto_traduzido": "limpo",
                            "confianca_ocr": 0.8,
                            "bbox": [5, 30, 10, 10],
                            "content_class": "sfx",
                        },
                    ]
                }
            ]
        },
    )
    _write_json(
        run / "debug" / "e2e" / "05_layout_geometry" / "bbox_coordinate_audit.json",
        {"summary": {"mixed_coordinate_space_count": 7}},
    )
    _write_json(
        run / "debug" / "e2e" / "08_inpaint" / "skip_inpaint_audit.json",
        {"summary": {"total_bands": 4, "bands_with_skip_honored": 3}},
    )
    (run / "pipeline.log").write_text(
        "[WARN] 2 text(s) sem balloon_bbox\n", encoding="utf-8"
    )

    report = module.analyze_root(root, write_report=True)
    metrics = report["runs"][0]["metrics"]

    assert metrics["exit_code"] == 0
    assert metrics["export_gate_status"] == "BLOCK"
    assert metrics["needs_review"] is True
    assert metrics["qa_export_consistent"] is False
    assert metrics["skip_inpaint_honored_bands"] == 3
    assert metrics["skip_inpaint_total_bands"] == 4
    assert metrics["mojibake_match_count"] == 1
    assert metrics["confianca_ocr_zero_count"] == 1
    assert metrics["mixed_coordinate_space_count"] == 7
    assert metrics["source_bbox_equals_balloon_bbox_count"] == 1
    assert metrics["balloon_bbox_missing_count"] == 2
    assert metrics["traduzai_env_vars"] == {
        "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1",
        "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1",
    }
    assert metrics["render_outside_count"] == 2
    assert metrics["render_on_art_count"] == 1
    assert metrics["bbox_overreach_count"] == 3
    assert metrics["content_class_counts"] == {"dialogue": 1, "sfx": 1}


# ---------------------------------------------------------------------------
# PR 9 — canonical counts, confidence, and content_class (DBG2-01/02/03)
# ---------------------------------------------------------------------------


def test_canonical_confidence_prefers_confidence_raw():
    module = _load_module()
    layer = {"confidence_raw": 0.812, "ocr_confidence": 0.0, "confianca_ocr": 0.0}
    assert module.canonical_confidence(layer) == 0.812


def test_canonical_confidence_missing_returns_none():
    module = _load_module()
    assert module.canonical_confidence({}) is None
    assert module.canonical_confidence({"confidence_raw": None}) is None
    assert module.canonical_confidence({"confidence_raw": True}) is None


def test_canonical_confidence_falls_back_to_legacy_fields():
    module = _load_module()
    assert module.canonical_confidence({"confianca_ocr": 0.74}) == 0.74
    assert module.canonical_confidence({"confidence": 0.5}) == 0.5


def test_canonical_source_does_not_double_text_layers_and_textos(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A_baseline_debug"
    run.mkdir(parents=True)

    duplicated_textos = [
        {
            "id": f"ocr_{i:03d}",
            "content_class": "dialogue",
            "confianca_ocr": 0.0,
            "ocr_confidence": 0.0,
        }
        for i in range(53)
    ] + [
        {
            "id": f"ocr_{i:03d}",
            "content_class": "narration",
            "confianca_ocr": 0.0,
        }
        for i in range(30)
    ] + [
        {"id": "tn1", "content_class": "tn_note"},
        {"id": "tn2", "content_class": "tn_note"},
        {"id": "noise", "content_class": "noise"},
        {"id": "sign", "content_class": "sign"},
        {"id": "url", "content_class": "url_watermark"},
    ]
    text_layers = [
        {**item, "confidence_raw": 0.7, "confianca_ocr": None, "ocr_confidence": None}
        for item in duplicated_textos
    ]
    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": text_layers,
                    "textos": duplicated_textos,
                }
            ],
            "estatisticas": {"total_textos": 88},
        },
    )

    report = module.analyze_run(run)
    metrics = report["metrics"]

    # 88 not 176 (no double counting).
    assert metrics["text_count"] == 88
    # content_class respects the canonical source (text_layers wins -> all
    # confidence_raw=0.7, so confidence_zero_count=0 even though textos has
    # confianca_ocr=0.0).
    assert metrics["confidence_zero_count"] == 0
    assert metrics["confidence_missing_count"] == 0
    assert metrics["content_class_counts"] == {
        "dialogue": 53,
        "narration": 30,
        "tn_note": 2,
        "noise": 1,
        "sign": 1,
        "url_watermark": 1,
    }


def test_confidence_zero_uses_ocr_audit_when_text_layers_only_have_legacy_zero(tmp_path):
    """If only the legacy ``confianca_ocr=0.0`` is present, we still trust the
    canonical confidence chain — but the OCR audit should align with the
    aggregator (cross-check). Regression for DBG2-01.
    """

    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A_baseline_debug"
    run.mkdir(parents=True)

    text_layers = [
        {
            "id": f"ocr_{i:03d}",
            "confidence_raw": 0.85,
            "ocr_confidence": None,
            "confianca_ocr": None,
            "content_class": "dialogue",
        }
        for i in range(88)
    ]
    _write_json(run / "project.json", {"paginas": [{"text_layers": text_layers}]})
    _write_json(
        run / "debug" / "e2e" / "03_ocr" / "ocr_confidence_audit.json",
        {"summary": {"blocks_with_confidence_zero": 0, "total_blocks": 88}},
    )

    report = module.analyze_run(run)
    metrics = report["metrics"]
    assert metrics["confidence_zero_count"] == 0
    assert metrics["confianca_ocr_zero_count"] == 0
    assert metrics["ocr_confidence_audit_zero_count"] == 0
    # Cross-check: stage-level and aggregator agree.
    consistency = metrics["debug_report_consistency"]
    confidence_check = next(
        c for c in consistency["checks"] if c["name"] == "confidence_zero_count"
    )
    assert confidence_check["consistent"] is True


# ---------------------------------------------------------------------------
# PR 9 — stage-level vs aggregator cross-check (DBG2-23)
# ---------------------------------------------------------------------------


def test_stage_file_overreach_count_is_canonical_when_present(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "confidence_raw": 0.8,
                            "source_bbox": [0, 0, 10, 10],
                            "balloon_bbox": [0, 0, 100, 100],
                            "content_class": "dialogue",
                        }
                    ]
                }
            ]
        },
    )
    # JSONL says there are TWO overreach rows; this is the canonical metric.
    _write_jsonl(
        run / "debug" / "e2e" / "05_layout_geometry" / "source_bbox_balloon_overreach.jsonl",
        [
            {"text_id": "ocr_001", "issue": "source_bbox_equals_balloon_bbox"},
            {"text_id": "ocr_002", "issue": "source_bbox_equals_balloon_bbox"},
        ],
    )

    metrics = module.analyze_run(run)["metrics"]
    consistency = metrics["debug_report_consistency"]
    overreach = next(
        c for c in consistency["checks"] if c["name"] == "source_bbox_overreach_count"
    )
    assert overreach["stage_level"] == 2
    assert overreach["aggregator"] == 2
    assert overreach["consistent"] is True
    assert metrics["source_bbox_equals_balloon_bbox_count"] == 2
    assert metrics["stage_file_aggregate_mismatch_count"] == 0


# ---------------------------------------------------------------------------
# PR 9 — skip_inpaint_honored aggregation (DBG2-21)
# ---------------------------------------------------------------------------


def test_stage_file_counts_are_used_as_aggregate_metrics_when_present(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "confidence_raw": 0.8,
                            "source_bbox": [0, 0, 10, 10],
                            "balloon_bbox": [0, 0, 100, 100],
                            "content_class": "dialogue",
                        }
                    ]
                }
            ]
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "05_layout_geometry" / "source_bbox_balloon_overreach.jsonl",
        [
            {"text_id": "ocr_001", "issue": "source_bbox_equals_balloon_bbox"},
            {"text_id": "ocr_002", "issue": "source_bbox_equals_balloon_bbox"},
        ],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "balloon_bbox_missing_audit.jsonl",
        [
            {"text_id": "ocr_003", "issue": "balloon_bbox_missing"},
            {"text_id": "ocr_004", "issue": "balloon_bbox_missing"},
            {"text_id": "ocr_005", "issue": "balloon_bbox_missing"},
        ],
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["source_bbox_equals_balloon_bbox_count"] == 2
    assert metrics["balloon_bbox_missing_count"] == 3
    assert metrics["stage_file_aggregate_mismatch_count"] == 0


def test_skip_inpaint_honored_aggregated_from_per_band_decisions(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A_baseline_debug"
    run.mkdir(parents=True)

    _write_json(run / "runner_config.json", {"skip_inpaint": False})
    # Three bands, none with skip honored (because not requested).
    for band in ("page_001_band_001", "page_001_band_002", "page_001_band_003"):
        _write_json(
            run / "debug" / "e2e" / "08_inpaint" / band / "inpaint_decision.json",
            {
                "band_id": band,
                "skip_inpaint_requested": False,
                "skip_inpaint_honored": False,
            },
        )

    metrics = module.analyze_run(run)["metrics"]
    assert metrics["skip_inpaint_total_bands"] == 3
    assert metrics["skip_inpaint_honored_bands"] == 0
    assert metrics["skip_inpaint_honored"] is False
    assert metrics["skip_inpaint_consistent"] is True


def test_skip_inpaint_honored_true_when_requested_and_no_inpaint_dir(tmp_path):
    """If config asked to skip and no per-band inpaint files exist, the
    aggregator should treat it as honored (B in the canonical ZIP).
    Resolves DBG2-21.
    """

    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "B_skip_inpaint"
    run.mkdir(parents=True)

    _write_json(run / "runner_config.json", {"skip_inpaint": True})
    # Create the dir but no per-band json.
    (run / "debug" / "e2e" / "08_inpaint").mkdir(parents=True, exist_ok=True)

    metrics = module.analyze_run(run)["metrics"]
    assert metrics["skip_inpaint_requested"] is True
    assert metrics["skip_inpaint_honored"] is True
    assert metrics["skip_inpaint_honored"] is not None
    assert metrics["skip_inpaint_consistent"] is True


def test_skip_inpaint_inconsistent_when_requested_but_some_band_skipped_not_honored(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "B"
    run.mkdir(parents=True)

    _write_json(run / "runner_config.json", {"skip_inpaint": True})
    _write_json(
        run / "debug" / "e2e" / "08_inpaint" / "page_001_band_001" / "inpaint_decision.json",
        {
            "band_id": "page_001_band_001",
            "skip_inpaint_requested": True,
            "skip_inpaint_honored": False,
        },
    )

    metrics = module.analyze_run(run)["metrics"]
    assert metrics["skip_inpaint_honored"] is False
    assert metrics["skip_inpaint_consistent"] is False


# ---------------------------------------------------------------------------
# PR 9 — render_plan introspection (DBG2-04/05)
# ---------------------------------------------------------------------------


def test_render_plan_metrics_count_null_ids_and_duplicates(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(run / "project.json", {"paginas": []})
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl",
        [
            {"text_id": "ocr_001", "page_id": "page_001", "band_id": "page_001_band_001", "coordinate_space": "page"},
            {"text_id": "ocr_001", "page_id": "page_001", "band_id": "page_001_band_001", "coordinate_space": "page"},
            {"text_id": "ocr_002", "page_id": None, "band_id": None, "coordinate_space": None},
            {"text_id": None, "page_id": None, "band_id": None, "coordinate_space": None},
        ],
    )

    metrics = module.analyze_run(run)["metrics"]
    assert metrics["render_plan_entry_count"] == 4
    assert metrics["render_plan_distinct_text_ids"] == 2
    assert metrics["render_plan_null_id_count"] >= 1
    assert metrics["render_plan_null_coordinate_space_count"] == 2
    assert metrics["render_plan_duplicate_final_entry_count"] == 1


def test_render_plan_metrics_detect_project_mismatch_and_trace_page_mismatch(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_002_band_003",
                            "page_id": "page_002",
                            "band_id": "page_002_band_003",
                            "balloon_bbox": [10, 20, 110, 120],
                            "safe_text_box": [20, 30, 100, 110],
                            "render_bbox": [30, 40, 90, 100],
                        }
                    ]
                }
            ]
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl",
        [
            {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_002_band_003",
                "page_id": "page_001",
                "band_id": "page_001_band_000",
                "coordinate_space": "page",
                "balloon_bbox": [10, 200, 110, 300],
                "safe_text_box": [20, 210, 100, 290],
                "render_bbox": [30, 220, 90, 280],
            }
        ],
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["render_plan_trace_page_mismatch_count"] == 1
    assert metrics["render_plan_project_mismatch_count"] == 1
    assert metrics["render_plan_project_field_mismatch_count"] >= 3


def test_render_plan_raw_pass_requires_project_skip_or_explicit_merge(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "ocr_001",
                            "trace_id": "ocr_001@page_003_band_048",
                            "band_id": "page_003_band_048",
                        },
                        {
                            "id": "ocr_merged",
                            "trace_id": "ocr_001@page_002_band_019",
                            "band_id": "page_002_band_019",
                            "source_trace_ids": ["ocr_002@page_002_band_019"],
                            "merge_reason": "same_balloon_merge",
                        },
                    ]
                }
            ]
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "render_plan_raw.jsonl",
        [
            {
                "text_id": "ocr_001",
                "trace_id": "ocr_001@page_003_band_048",
                "page_id": "page_003",
                "band_id": "page_003_band_048",
                "fit_status": "PASS",
            },
            {
                "text_id": "ocr_002",
                "trace_id": "ocr_002@page_002_band_019",
                "page_id": "page_002",
                "band_id": "page_002_band_019",
                "fit_status": "PASS",
            },
            {
                "text_id": "ocr_003",
                "trace_id": "ocr_003@page_004_band_010",
                "page_id": "page_004",
                "band_id": "page_004_band_010",
                "fit_status": "PASS",
                "original": "I SHOULD NOT DISAPPEAR",
            },
        ],
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["render_plan_raw_pass_distinct_trace_count"] == 3
    assert metrics["render_plan_raw_pass_project_trace_count"] == 1
    assert metrics["render_plan_raw_pass_explicit_merge_count"] == 1
    assert metrics["render_plan_raw_pass_missing_project_or_skip_count"] == 1
    assert metrics["render_plan_raw_pass_missing_examples"][0]["trace_id"] == "ocr_003@page_004_band_010"


def test_project_render_geometry_counts_render_outside_final_balloon(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "inside",
                            "trace_id": "inside@page_001_band_001",
                            "band_id": "page_001_band_001",
                            "balloon_bbox": [10, 10, 100, 100],
                            "render_bbox": [20, 20, 90, 90],
                        },
                        {
                            "id": "outside",
                            "trace_id": "outside@page_001_band_002",
                            "band_id": "page_001_band_002",
                            "balloon_bbox": [10, 10, 100, 100],
                            "render_bbox": [20, 120, 90, 180],
                        },
                    ]
                }
            ]
        },
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["project_render_geometry_total_count"] == 2
    assert metrics["project_render_outside_count"] == 1
    assert metrics["render_outside_count"] == 1


def test_qa_issue_traceability_missing_is_reported(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(run / "project.json", {"paginas": []})
    _write_jsonl(
        run / "debug" / "e2e" / "11_qa_export_gate" / "qa_issues.jsonl",
        [
            {"type": "p0_render_blocker", "severity": "critical", "layer": "ocr_001"},
            {
                "type": "p0_render_blocker",
                "severity": "critical",
                "layer": "ocr_002",
                "trace_id": "ocr_002@page_001_band_002",
                "text_id": "ocr_002",
                "page_id": "page_001",
                "band_id": "page_001_band_002",
            },
        ],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "11_qa_export_gate" / "visual_blockers.jsonl",
        [{"type": "p0_render_blocker", "severity": "critical", "layer": "ocr_001"}],
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["qa_issue_traceability_missing_count"] == 1
    assert metrics["visual_blocker_traceability_missing_count"] == 1


# ---------------------------------------------------------------------------
# PR 17 — strict gate
# ---------------------------------------------------------------------------


def test_pr17_metrics_cover_audit_and_project_layer_invariants(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "joined",
                            "normalization_trace": {"changed": True, "normalized": "CAN YOU"},
                        },
                        {
                            "id": "noise",
                            "content_class": "noise",
                            "tipo": "fala",
                            "skip_processing": False,
                        },
                        {
                            "id": "sign",
                            "content_class": "sign",
                            "tipo": "narracao",
                        },
                    ]
                }
            ]
        },
    )
    _write_json(
        run / "debug" / "e2e" / "05_layout_geometry" / "bbox_coordinate_audit.json",
        {"summary": {"derived_bbox_coordinate_mismatch_count": 2}},
    )
    _write_json(
        run / "debug" / "e2e" / "11_qa_export_gate" / "qa_flag_propagation_audit.json",
        {"summary": {"qa_flag_not_propagated_count": 1}},
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["derived_bbox_coordinate_mismatch_count"] == 2
    assert metrics["qa_flag_not_propagated_count"] == 1
    assert metrics["normalized_text_not_propagated_count"] == 1
    assert metrics["cover_noise_rendered_as_dialogue_count"] == 1
    assert metrics["sign_rendered_as_narration_count"] == 1


def test_evaluate_invariants_passes_on_clean_run(tmp_path):
    module = _load_module()
    report = {
        "runs": [
            {
                "name": "A",
                "metrics": {
                    "debug_report_metric_mismatch_count": 0,
                    "trace_id_null_count": 0,
                    "project_textos_trace_id_null_count": 0,
                    "project_trace_id_unique_ratio": 1.0,
                    "page_band_mismatch_count": 0,
                    "qa_export_consistent": True,
                    "translation_summary_mismatch": False,
                    "translation_input_null_trace_id_count": 0,
                    "translation_output_null_trace_id_count": 0,
                    "translation_input_null_band_id_count": 0,
                    "translation_output_null_band_id_count": 0,
                    "translation_pair_missing_outputs_count": 0,
                    "translation_pair_orphan_outputs_count": 0,
                    "render_plan_null_id_count": 0,
                    "render_plan_null_trace_id_count": 0,
                    "render_plan_final_incomplete": False,
                    "render_plan_null_coordinate_space_count": 0,
                    "render_plan_duplicate_final_entry_count": 0,
                    "derived_bbox_coordinate_mismatch_count": 0,
                    "inpaint_debug_missing": False,
                    "inpaint_trace_id_missing_count": 0,
                    "copyback_trace_ids_missing_count": 0,
                    "copyback_band_id_missing_count": 0,
                    "source_bbox_equals_balloon_bbox_count": 0,
                    "detect_accepted_null_match_count": 0,
                    "debug_errors_count": 0,
                    "translation_debug_entry_count": 1,
                    "translation_debug_missing": False,
                    "copyback_debug_missing": False,
                    "translated_comparison_present": True,
                    "qa_flag_not_propagated_count": 0,
                    "normalized_text_not_propagated_count": 0,
                    "cover_noise_rendered_as_dialogue_count": 0,
                    "sign_rendered_as_narration_count": 0,
                    "skip_inpaint_consistent": True,
                    "stage_file_aggregate_mismatch_count": 0,
                    "text_count": 10,
                },
            }
        ]
    }
    result = module.evaluate_invariants(report)
    assert result["all_passed"] is True


def test_evaluate_invariants_fails_when_any_metric_breaks(tmp_path):
    module = _load_module()
    report = {
        "runs": [
            {
                "name": "B",
                "metrics": {
                    "debug_report_metric_mismatch_count": 1,
                    "render_plan_null_id_count": 0,
                    "render_plan_null_coordinate_space_count": 1,
                    "render_plan_duplicate_final_entry_count": 1,
                    "derived_bbox_coordinate_mismatch_count": 1,
                    "translation_debug_entry_count": 0,
                    "translation_debug_missing": True,
                    "copyback_debug_missing": True,
                    "translated_comparison_present": False,
                    "qa_flag_not_propagated_count": 2,
                    "normalized_text_not_propagated_count": 3,
                    "cover_noise_rendered_as_dialogue_count": 1,
                    "sign_rendered_as_narration_count": 1,
                    "skip_inpaint_consistent": False,
                    "stage_file_aggregate_mismatch_count": 2,
                    "text_count": 88,
                },
            }
        ]
    }
    result = module.evaluate_invariants(report)
    assert result["all_passed"] is False
    failed_names = {item["name"] for item in result["invariants"] if not item["passed"]}
    assert "debug_report_metric_mismatch_count == 0" in failed_names
    assert "render_plan_null_coordinate_space_count == 0" in failed_names
    assert "render_plan_duplicate_final_entry_count == 0" in failed_names
    assert "derived_bbox_coordinate_mismatch_count == 0" in failed_names
    assert "qa_flag_not_propagated_count == 0" in failed_names
    assert "normalized_text_not_propagated_count == 0" in failed_names
    assert "cover_noise_rendered_as_dialogue_count == 0" in failed_names
    assert "sign_rendered_as_narration_count == 0" in failed_names
    assert "skip_inpaint_honored_consistent" in failed_names
    assert "contact_sheets_translated_comparison_present" in failed_names


def test_v2_identity_and_stage_trace_metrics_surface_current_gaps(tmp_path):
    module = _load_module()
    run = tmp_path / "A"
    run.mkdir()

    _write_json(
        run / "project.json",
        {
                "paginas": [
                    {
                        "text_layers": [
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_001_band_001",
                            "band_id": "page_001_band_001",
                            "render_bbox": [1, 2, 3, 4],
                        },
                        {
                            "id": "ocr_001",
                            "text_id": "ocr_001",
                            "trace_id": "ocr_001@page_001_band_002",
                            "band_id": "page_001_band_002",
                            "render_bbox": [5, 6, 7, 8],
                        },
                    ],
                    "textos": [{"id": "ocr_001"}, {"id": "ocr_001"}],
                }
            ]
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "07_translation" / "translation_inputs.jsonl",
        [
            {"page_id": "page_001", "text_id": "ocr_001", "prompt_hash": "a"},
            {"page_id": "page_001", "text_id": "ocr_001", "prompt_hash": "b"},
        ],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "07_translation" / "translation_outputs.jsonl",
        [{"page_id": "page_001", "text_id": "ocr_001", "prompt_hash": "a"}],
    )
    _write_json(
        run / "debug" / "e2e" / "07_translation" / "translation_debug_summary.json",
        {"translation_inputs_count": 1, "translation_outputs_count": 1},
    )
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl",
        [
            {
                "text_id": "ocr_001",
                "page_id": "page_001",
                "band_id": "page_001_band_001",
                "coordinate_space": "page",
            }
        ],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "08_inpaint" / "inpaint_blocks.jsonl",
        [{"text_id": "ocr_001"}],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "10_copyback_reassemble" / "copyback_decisions.jsonl",
        [{"band_id": "page_001_band_001", "text_count": 1}],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "02_strip_detect" / "detect_candidates.jsonl",
        [{"accepted": True, "matched_text_id": None}],
    )

    metrics = module.analyze_run(run)["metrics"]

    assert metrics["trace_id_null_count"] == 0
    assert metrics["project_textos_trace_id_null_count"] == 2
    assert metrics["project_trace_id_unique_ratio"] == 1.0
    assert metrics["text_id_duplicate_count"] == 1
    assert metrics["translation_summary_mismatch"] is True
    assert metrics["translation_input_null_trace_id_count"] == 2
    assert metrics["translation_input_null_band_id_count"] == 2
    assert metrics["translation_pair_missing_outputs_count"] == 1
    assert metrics["render_plan_null_trace_id_count"] == 1
    assert metrics["render_plan_final_incomplete"] is True
    assert metrics["render_plan_missing_final_entry_count"] == 1
    assert metrics["inpaint_trace_id_missing_count"] == 1
    assert metrics["copyback_trace_ids_missing_count"] == 1
    assert metrics["detect_accepted_null_match_count"] == 1


def test_main_exit_code_3_on_strict_audit_failure(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(run / "project.json", {"paginas": []})

    rc = module.main([str(root), "--strict-debug-audit", "--write-report"])
    assert rc == 3
    # The written report includes the strict audit payload.
    written = json.loads((root / "debug_report.json").read_text(encoding="utf-8"))
    assert written["strict_audit"]["all_passed"] is False


def test_main_exit_code_0_when_invariants_all_pass(tmp_path):
    module = _load_module()
    root = tmp_path / "debug" / "e2e"
    run = root / "A"
    run.mkdir(parents=True)

    _write_json(
        run / "project.json",
        {
            "paginas": [
                {
                    "text_layers": [
                        {
                            "id": "a",
                            "text_id": "a",
                            "trace_id": "a@page_001_band_001",
                            "band_id": "page_001_band_001",
                            "confidence_raw": 0.7,
                                "content_class": "dialogue",
                            }
                        ],
                        "textos": [
                            {
                                "id": "a",
                                "text_id": "a",
                                "trace_id": "a@page_001_band_001",
                                "page_id": "page_001",
                                "band_id": "page_001_band_001",
                            }
                        ],
                    }
                ],
            "estatisticas": {"total_textos": 1},
        },
    )
    _write_json(run / "runner_config.json", {"skip_inpaint": False})
    _write_json(
        run / "debug" / "e2e" / "11_qa_export_gate" / "qa_export_gate_consistency.json",
        {
            "summary": {"critical_count": 0},
            "export_gate": {"status": "PASS", "critical_issue_count": 0},
            "consistent": True,
        },
    )
    _write_jsonl(
        run / "debug" / "e2e" / "07_translation" / "translation_inputs.jsonl",
        [
            {
                "trace_id": "a@page_001_band_001",
                "page_id": "page_001",
                "band_id": "page_001_band_001",
                "text_id": "a",
                "source_text": "hello",
            }
        ],
    )
    _write_jsonl(
        run / "debug" / "e2e" / "07_translation" / "translation_outputs.jsonl",
        [
            {
                "trace_id": "a@page_001_band_001",
                "page_id": "page_001",
                "band_id": "page_001_band_001",
                "text_id": "a",
                "translated_text": "ola",
            }
        ],
    )
    _write_json(
        run / "debug" / "e2e" / "07_translation" / "translation_debug_summary.json",
        {"translation_inputs_count": 1, "translation_outputs_count": 1},
    )
    # Copyback + contact sheet artefacts.
    copyback_path = run / "debug" / "e2e" / "10_copyback_reassemble" / "copyback_decisions.jsonl"
    copyback_path.parent.mkdir(parents=True, exist_ok=True)
    copyback_path.write_text(
        json.dumps(
            {
                "band_id": "page_001_band_001",
                "text_count": 1,
                "trace_ids_in_band": ["a@page_001_band_001"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contact_path = run / "debug" / "e2e" / "12_contact_sheets" / "translated_comparison.jpg"
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path.write_bytes(b"\xff\xd8\xff\xe0fake")
    # Render plan: single final entry, all IDs populated.
    _write_jsonl(
        run / "debug" / "e2e" / "09_typeset" / "render_plan_final.jsonl",
        [
            {
                "text_id": "a",
                "trace_id": "a@page_001_band_001",
                "page_id": "page_001",
                "band_id": "page_001_band_001",
                "coordinate_space": "page",
            }
        ],
        )

    _write_jsonl(
        run / "debug" / "e2e" / "08_inpaint" / "inpaint_blocks.jsonl",
        [
            {
                "band_id": "page_001_band_001",
                "page_id": "page_001",
                "trace_ids": ["a@page_001_band_001"],
            }
        ],
    )

    rc = module.main([str(root), "--strict-debug-audit", "--write-report"])
    assert rc == 0
